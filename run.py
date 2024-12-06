import os
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics import YOLO

from functions import PatchTransformer
from functions import PatchApplier
from functions import MaxProbExtractor
from functions import SaliencyLoss
from functions import NPSLoss
from functions import TotalVariationLoss
from functions import YOLODataset
from functions import YOLOAnnotationTransformer
from convert_visdrone_to_yolo import conv_visdrone_2_yolo

# Configuration Class
class cfg:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Transformation
        self.target_size_frac = [0.25, 0.40]
        self.mul_gau_mean = [0.5, 0.8]
        self.mul_gau_std = 0.1
        self.x_off_loc = [-0.25, 0.25]
        self.y_off_loc = [-0.25, 0.25]

        self.use_mul_add_gau = True
        self.transform_patches = True
        self.rotate_patches = True
        self.random_patch_loc = True

        self.sal_mult = 1
        self.nps_mult = 2.5
        self.tv_mult = 0

        self.patch_pixel_range = [0, 255]

        # Patch
        self.patch_alpha = 1
        self.patch_size = [64, 64]

        # Yolo Dataset
        self.image_dir = "./dataset/visdrone_train/VisDrone2019-DET-train/images"
        self.label_dir = "./dataset/visdrone_train/VisDrone2019-DET-train/yolo_annotations"
        self.max_labels = 48
        self.model_in_sz = [640, 640]
        self.use_even_odd_images = "all"
        self.objective_class_id = None
        self.min_pixel_area = None

        # Visual Loss
        self.triplet_printfile = "30_rgb_triplets.csv"

        # Training Loss
        self.loss_target = "obj*cls"

        # Training
        self.batch_size = 2
        self.n_epochs = 420
        self.start_lr = 0.03
        self.min_tv_loss = 0.1
        self.n_classes = 4

        # Logs
        self.log_dir = "./logs/"
        self.patch_save_epoch_freq = 1
        self.tensorboard_port = 8080
        self.patch_name = "base"


# Trainer Class
class myTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = cfg.device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval()

        self.patch_transformer = PatchTransformer(
            cfg.target_size_frac,
            cfg.mul_gau_mean,
            cfg.mul_gau_std,
            cfg.x_off_loc,
            cfg.y_off_loc,
            self.dev).to(self.dev)

        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)
        self.prob_extractor = MaxProbExtractor(cfg).to(self.dev)
        self.sal_loss = SaliencyLoss().to(self.dev)
        self.nps_loss = NPSLoss(cfg.triplet_printfile, cfg.patch_size).to(self.dev)
        self.tv_loss = TotalVariationLoss().to(self.dev)

        # Freeze entire detection model
        for param in self.model.parameters():
            param.requires_grad = False

        # Load training dataset
        self.YOLODataset = YOLODataset(
            image_dir=cfg.image_dir,
            label_dir=cfg.label_dir,
            max_labels=cfg.max_labels,
            model_in_sz=cfg.model_in_sz,
            use_even_odd_images=cfg.use_even_odd_images,
            transform=None,
            filter_class_ids=cfg.objective_class_id,
            min_pixel_area=cfg.min_pixel_area,
            shuffle=True,
        )

        self.train_loader = DataLoader(
            self.YOLODataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.dev.type == "cuda" else False,
        )

        self.epoch_length = len(self.train_loader)

    def save_checkpoint(self, epoch, adv_patch_cpu, optimizer, scheduler, checkpoint_path):
        checkpoint = {
            "epoch": epoch,
            "adv_patch_cpu": adv_patch_cpu,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None, None, None, 1  # Default to starting at epoch 1

        checkpoint = torch.load(checkpoint_path, map_location=self.dev)
        print(f"Checkpoint loaded: {checkpoint_path}")

        adv_patch_cpu = checkpoint["adv_patch_cpu"]
        adv_patch_cpu.requires_grad = True

        optimizer = optim.Adam([adv_patch_cpu], lr=self.cfg.start_lr, amsgrad=True)
        optimizer.load_state_dict(checkpoint["optimizer"])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"] + 1
        return adv_patch_cpu, optimizer, scheduler, start_epoch

    def train(self):
        patch_dir = os.path.join(self.cfg.log_dir, "patches")
        os.makedirs(patch_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.cfg.log_dir, "checkpoint.pth")

        loss_target = self.cfg.loss_target
        if loss_target == "obj":
            self.cfg.loss_target = lambda obj, cls: obj
        elif loss_target == "cls":
            self.cfg.loss_target = lambda obj, cls: cls
        elif loss_target in {"obj * cls", "obj*cls"}:
            self.cfg.loss_target = lambda obj, cls: obj * cls
        else:
            raise NotImplementedError(f"Loss target {loss_target} not been implemented")

        adv_patch_cpu, optimizer, scheduler, start_epoch = self.load_checkpoint(checkpoint_path)

        if adv_patch_cpu is None:
            p_c = 3
            p_w, p_h = self.cfg.patch_size
            adv_patch_cpu = torch.rand((p_c, p_h, p_w))
            adv_patch_cpu.requires_grad = True
            optimizer = optim.Adam([adv_patch_cpu], lr=self.cfg.start_lr, amsgrad=True)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)

        start_time = time.time()
        for epoch in range(start_epoch, self.cfg.n_epochs + 1):
            ep_loss = 0
            min_tv_loss = torch.tensor(self.cfg.min_tv_loss, device=self.dev)
            zero_tensor = torch.tensor([0], device=self.dev)

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(self.train_loader), desc=f"Train epoch {epoch}", total=self.epoch_length):
                img_batch = img_batch.to(self.dev, non_blocking=True)
                lab_batch = lab_batch.to(self.dev, non_blocking=True)
                adv_patch = adv_patch_cpu.to(self.dev, non_blocking=True)

                adv_batch_t = self.patch_transformer(
                    adv_patch,
                    lab_batch,
                    self.cfg.model_in_sz,
                    use_mul_add_gau=self.cfg.use_mul_add_gau,
                    do_transforms=self.cfg.transform_patches,
                    do_rotate=self.cfg.rotate_patches,
                    rand_loc=self.cfg.random_patch_loc,
                )
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch, (self.cfg.model_in_sz[0], self.cfg.model_in_sz[1]))

                output = self.model(p_img_batch)[0]
                max_prob = self.prob_extractor(output)
                sal = self.sal_loss(adv_patch) if self.cfg.sal_mult != 0 else zero_tensor
                nps = self.nps_loss(adv_patch) if self.cfg.nps_mult != 0 else zero_tensor
                tv = self.tv_loss(adv_patch) if self.cfg.tv_mult != 0 else zero_tensor

                det_loss = torch.mean(max_prob)
                sal_loss = sal * self.cfg.sal_mult
                nps_loss = nps * self.cfg.nps_mult
                tv_loss = torch.max(tv * self.cfg.tv_mult, min_tv_loss)

                loss = det_loss + sal_loss + nps_loss + tv_loss
                ep_loss += loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                pl, ph = self.cfg.patch_pixel_range
                adv_patch_cpu.data.clamp_(pl / 255, ph / 255)

            ep_loss = ep_loss / len(self.train_loader)
            scheduler.step(ep_loss)

            self.save_checkpoint(epoch, adv_patch_cpu, optimizer, scheduler, checkpoint_path)
            print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s, Loss: {ep_loss}")


if __name__ == "__main__":
    mycfg = cfg()
    trainer = myTrainer(mycfg)
    trainer.train()