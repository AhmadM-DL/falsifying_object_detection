import torch
import torchvision
from torch import nn
from typing import Union, Tuple, Optional
import math, os
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import time

from torchvision import transforms as T
from torch import autograd, optim
from tqdm import tqdm

import glob
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MedianPool2d(nn.Module):
    """
    Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
    
    
class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    """

    def __init__(
        self,
        t_size_frac: Union[float, Tuple[float, float]] = 0.3,
        mul_gau_mean: Union[float, Tuple[float, float]] = (0.5, 0.8),
        mul_gau_std: Union[float, Tuple[float, float]] = 0.1,
        x_off_loc: Tuple[float, float] = [-0.25, 0.25],
        y_off_loc: Tuple[float, float] = [-0.25, 0.25],
        dev: torch.device = torch.device("cuda:0"),
    ):
        super(PatchTransformer, self).__init__()
        # convert to duplicated lists/tuples to unpack and send to np.random.uniform
        self.t_size_frac = [t_size_frac, t_size_frac] if isinstance(t_size_frac, float) else t_size_frac
        self.m_gau_mean = [mul_gau_mean, mul_gau_mean] if isinstance(mul_gau_mean, float) else mul_gau_mean
        self.m_gau_std = [mul_gau_std, mul_gau_std] if isinstance(mul_gau_std, float) else mul_gau_std
        assert (
            len(self.t_size_frac) == 2 and len(self.m_gau_mean) == 2 and len(self.m_gau_std) == 2
        ), "Range must have 2 values"
        self.x_off_loc = x_off_loc
        self.y_off_loc = y_off_loc
        self.dev = dev
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(kernel_size=7, same=True)

        self.tensor = torch.FloatTensor if "cpu" in str(dev) else torch.cuda.FloatTensor

    def forward(
        self, adv_patch, lab_batch, model_in_sz, use_mul_add_gau=True, do_transforms=True, do_rotate=True, rand_loc=True
    ):
        # add gaussian noise to reduce contrast with a stohastic process
        p_c, p_h, p_w = adv_patch.shape
        if use_mul_add_gau:
            mul_gau = torch.normal(
                np.random.uniform(*self.m_gau_mean),
                np.random.uniform(*self.m_gau_std),
                (p_c, p_h, p_w),
                device=self.dev,
            )
            add_gau = torch.normal(0, 0.001, (p_c, p_h, p_w), device=self.dev)
            adv_patch = adv_patch * mul_gau + add_gau
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        m_h, m_w = model_in_sz
        # Determine size of padding
        pad = (m_w - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1
        )  # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms
        if do_transforms:
            # Create random contrast tensor
            contrast = self.tensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random brightness tensor
            brightness = self.tensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random noise tensor
            noise = self.tensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

            # Apply contrast/brightness/noise, clamp
            adv_batch = adv_batch * contrast + brightness + noise

            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = lab_batch[..., 0].unsqueeze(-1)  # equiv to torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, p_c)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = self.tensor(cls_mask.size()).fill_(1)

        # Pad patch and mask to image dimensions
        patch_pad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = patch_pad(adv_batch)
        msk_batch = patch_pad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = lab_batch.size(0) * lab_batch.size(1)
        if do_rotate:
            angle = self.tensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = self.tensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = self.tensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * m_w
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * m_w
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * m_w
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * m_w
        tsize = np.random.uniform(*self.t_size_frac)
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(tsize)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(tsize)) ** 2)
        )

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if rand_loc:
            off_x = targetoff_x * (self.tensor(targetoff_x.size()).uniform_(*self.x_off_loc))
            target_x = target_x + off_x
            off_y = targetoff_y * (self.tensor(targetoff_y.size()).uniform_(*self.x_off_loc))
            target_y = target_y + off_y
        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation/rescale matrix
        # Theta = input batch of affine matrices with shape (N×2×3) for 2D or (N×3×4) for 3D
        theta = self.tensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        return adv_batch_t * msk_batch_t
    
    
class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    The patch (adv_batch) has the same size as the image, just is zero everywhere there isn't a patch.
    If patch_alpha == 1 (default), just overwrite the background image values with the patch values.
    Else, blend the patch with the image
    See: https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
         https://stackoverflow.com/questions/49737541/merge-two-images-with-alpha-channel/49738078
        I = \alpha F + (1 - \alpha) B
            F = foregraound (patch, or adv_batch)
            B = background (image, or img_batch)
    """

    def __init__(self, patch_alpha: float = 1):
        super(PatchApplier, self).__init__()
        self.patch_alpha = patch_alpha

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            # replace image values with patch values
            if self.patch_alpha == 1:
                img_batch = torch.where((adv == 0), img_batch, adv)
            # alpha blend
            else:
                # get combo of image and adv
                alpha_blend = self.patch_alpha * adv + (1.0 - self.patch_alpha) * img_batch
                # apply alpha blend where the patch is non-zero
                img_batch = torch.where((adv == 0), img_batch, alpha_blend)

        return img_batch
    
    
class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, config):
        super(MaxProbExtractor, self).__init__()
        self.config = config

    def forward(self, output: torch.Tensor):
        """Output must be of the shape [batch, -1, 5 + num_cls]"""
        # get values necessary for transformation
        assert output.size(-1) == (5 + self.config.n_classes)

        class_confs = output[:, :, 5 : 5 + self.config.n_classes]  # [batch, -1, n_classes]
        objectness_score = output[:, :, 4]  # [batch, -1, 5 + num_cls] -> [batch, -1], no need to run sigmoid here

        if self.config.objective_class_id is not None:
            # norm probs for object classes to [0, 1]
            class_confs = torch.nn.Softmax(dim=2)(class_confs)
            # only select the conf score for the objective class
            class_confs = class_confs[:, :, self.config.objective_class_id]
        else:
            # get class with highest conf for each box if objective_class_id is None
            class_confs = torch.max(class_confs, dim=2)[0]  # [batch, -1, 4] -> [batch, -1]

        confs_if_object = self.config.loss_target(objectness_score, class_confs)
        max_conf, _ = torch.max(confs_if_object, dim=1)
        return max_conf
    
    
    
    
class SaliencyLoss(nn.Module):
    """
    Implementation of the colorfulness metric as the saliency loss.

    The smaller the value, the less colorful the image.
    Reference: https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf
    """

    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Float Tensor of shape [C, H, W] where C=3 (R, G, B channels)
        """
        assert adv_patch.shape[0] == 3
        r, g, b = adv_patch
        rg = r - g
        yb = 0.5 * (r + g) - b

        mu_rg, sigma_rg = torch.mean(rg) + 1e-8, torch.std(rg) + 1e-8
        mu_yb, sigma_yb = torch.mean(yb) + 1e-8, torch.std(yb) + 1e-8
        sl = torch.sqrt(sigma_rg**2 + sigma_yb**2) + (0.3 * torch.sqrt(mu_rg**2 + mu_yb**2))
        return sl / torch.numel(adv_patch)
    
    
    

class TotalVariationLoss(nn.Module):
    """TotalVariationLoss: calculates the total variation of a patch.
    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.
    Reference: https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W]
        """
        # calc diff in patch rows
        tvcomp_r = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), dim=0)
        tvcomp_r = torch.sum(torch.sum(tvcomp_r, dim=0), dim=0)
        # calc diff in patch columns
        tvcomp_c = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), dim=0)
        tvcomp_c = torch.sum(torch.sum(tvcomp_c, dim=0), dim=0)
        tv = tvcomp_r + tvcomp_c
        return tv / torch.numel(adv_patch)
    
    
class NPSLoss(nn.Module):
    """NMSLoss: calculates the non-printability-score loss of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    However, a summation of the differences is used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
    """

    def __init__(self, triplet_scores_fpath: str, size: Tuple[int, int]):
        super(NPSLoss, self).__init__()
        self.printability_array = nn.Parameter(
            self.get_printability_array(triplet_scores_fpath, size), requires_grad=False
        )

    def forward(self, adv_patch):
        # calculate euclidean distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = adv_patch - self.printability_array + 0.000001
        color_dist = color_dist**2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # use the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, triplet_scores_fpath: str, size: Tuple[int, int]) -> torch.Tensor:
        """
        Get printability tensor array holding the rgb triplets (range [0,1]) loaded from triplet_scores_fpath
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
        """
        ref_triplet_list = []
        # read in reference printability triplets into a list
        with open(triplet_scores_fpath, "r", encoding="utf-8") as f:
            for line in f:
                ref_triplet_list.append(line.strip().split(","))

        p_h, p_w = size
        printability_array = []
        for ref_triplet in ref_triplet_list:
            r, g, b = map(float, ref_triplet)
            ref_tensor_img = torch.stack(
                [torch.full((p_h, p_w), r), torch.full((p_h, p_w), g), torch.full((p_h, p_w), b)]
            )
            printability_array.append(ref_tensor_img.float())
        return torch.stack(printability_array)
    
    
    
IMG_EXTNS = {".png", ".jpg", ".jpeg"}

class YOLODataset(Dataset):
    """
    Create a dataset for adversarial-yolt.

    Attributes:
        image_dir: Directory containing the images of the YOLO format dataset.
        label_dir: Directory containing the labels of the YOLO format dataset.
        max_labels: max number labels to use for each image
        model_in_sz: model input image size (height, width)
        use_even_odd_images: optionally load a data subset based on the last numeric char of the img filename [all, even, odd]
        filter_class_id: np.ndarray class id(s) to get. Set None to get all classes
        min_pixel_area: min pixel area below which all boxes are filtered out. (Out of the model in size area)
        shuffle: Whether or not to shuffle the dataset.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        max_labels: int,
        model_in_sz: Tuple[int, int],
        use_even_odd_images: str = "all",
        transform: Optional[torch.nn.Module] = None,
        filter_class_ids: Optional[np.array] = None,
        min_pixel_area: Optional[int] = None,
        shuffle: bool = True,
    ):
        assert use_even_odd_images in {"all", "even", "odd"}, "use_even_odd param can only be all, even or odd"
        image_paths = glob.glob(osp.join(image_dir, "*"))
        label_paths = glob.glob(osp.join(label_dir, "*"))
        image_paths = sorted([p for p in image_paths if osp.splitext(p)[-1] in IMG_EXTNS])
        label_paths = sorted([p for p in label_paths if osp.splitext(p)[-1] in {".txt"}])

        # if use_even_odd_images is set, use images with even/odd numbers in the last char of their filenames
        if use_even_odd_images in {"even", "odd"}:
            rem = 0 if use_even_odd_images == "even" else 1
            image_paths = [p for p in image_paths if int(osp.splitext(p)[0][-1]) % 2 == rem]
            label_paths = [p for p in label_paths if int(osp.splitext(p)[0][-1]) % 2 == rem]
        assert len(image_paths) == len(label_paths), "Number of images and number of labels don't match"
        # all corresponding image and labels must exist
        for img, lab in zip(image_paths, label_paths):
            if osp.basename(img).split(".")[0] != osp.basename(lab).split(".")[0]:
                raise FileNotFoundError(f"Matching image {img} or label {lab} not found")
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.model_in_sz = model_in_sz
        self.shuffle = shuffle
        self.max_n_labels = max_labels
        self.transform = transform
        self.filter_class_ids = np.asarray(filter_class_ids) if filter_class_ids is not None else None
        self.min_pixel_area = min_pixel_area

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert idx <= len(self), "Index range error"
        img_path = self.image_paths[idx]
        lab_path = self.label_paths[idx]
        image = Image.open(img_path).convert("RGB")
        # check to see if label file contains any annotation data
        label = np.loadtxt(lab_path) if osp.getsize(lab_path) else np.zeros([1, 5])
        if label.ndim == 1:
            label = np.expand_dims(label, axis=0)
        # sort in reverse by bbox area
        label = np.asarray(sorted(label, key=lambda annot: -annot[3] * annot[4]))
        # selectively get classes if filter_class_ids is not None
        if self.filter_class_ids is not None:
            label = label[np.isin(label[:, 0], self.filter_class_ids)]
            label = label if len(label) > 0 else np.zeros([1, 5])

        label = torch.from_numpy(label).float()
        image, label = self.pad_and_scale(image, label)
        if self.transform:
            image = self.transform(image)
            if np.random.random() < 0.5:  # rand horizontal flip
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if label.shape:
                    label[:, 1] = 1 - label[:, 1]
        # filter boxes by bbox area pixels compared to the model in size (640x640 by default)
        if self.min_pixel_area is not None:
            label = label[
                (label[:, 3] * label[:, 4]) >= (self.min_pixel_area / (self.model_in_sz[0] * self.model_in_sz[1]))
            ]
            label = label if len(label) > 0 else torch.zeros([1, 5])
        image = transforms.ToTensor()(image)
        label = self.pad_label(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """Pad image and adjust label img is a PIL image lab is of fmt class x_center y_center width height with
        normalized coords.
        """
        img_w, img_h = img.size
        if img_w == img_h:
            padded_img = img
        else:
            if img_w < img_h:
                padding = (img_h - img_w) / 2
                padded_img = Image.new("RGB", (img_h, img_h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * img_w + padding) / img_h
                lab[:, [3]] = lab[:, [3]] * img_w / img_h
            else:
                padding = (img_w - img_h) / 2
                padded_img = Image.new("RGB", (img_w, img_w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * img_h + padding) / img_w
                lab[:, [4]] = lab[:, [4]] * img_h / img_w
        padded_img = transforms.Resize(self.model_in_sz)(padded_img)

        return padded_img, lab

    def pad_label(self, label: torch.Tensor) -> torch.Tensor:
        """Pad labels with zeros if fewer labels than max_n_labels present."""
        pad_size = self.max_n_labels - label.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(label, (0, 0, 0, pad_size), value=0)
        else:
            padded_lab = label[: self.max_n_labels]
        return padded_lab
    
    
# import os

# class YOLOAnnotationTransformer:
#     def __init__(self, class_id=0):
#         """
#         Initialize the YOLOAnnotationTransformer class.

#         Args:
#             class_id (int): Class ID for the object annotations (default: 0).
#         """
#         self.class_id = class_id

#     def convert_to_yolo(self, image_width, image_height, bbox):
#         """
#         Convert bounding box from (x_min, y_min, x_max, y_max) to YOLO format.

#         Args:
#             image_width (int): Width of the image.
#             image_height (int): Height of the image.
#             bbox (tuple): Bounding box in pixel coordinates (x_min, y_min, x_max, y_max).

#         Returns:
#             str: YOLO-formatted annotation (class_id center_x center_y width height).
#         """
#         x_min, y_min, x_max, y_max = bbox
#         center_x = (x_min + x_max) / 2 / image_width
#         center_y = (y_min + y_max) / 2 / image_height
#         width = (x_max - x_min) / image_width
#         height = (y_max - y_min) / image_height
#         return f"{self.class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"

#     def transform_annotations_to_yolo(self, annotation_file, output_dir, image_width, image_height):
#         """
#         Transform annotations from pixel format to YOLO format and save to file.

#         Args:
#             annotation_file (str): Path to the annotation file containing bounding boxes.
#             output_dir (str): Directory to save YOLO-format annotation files.
#             image_width (int): Width of the corresponding image.
#             image_height (int): Height of the corresponding image.
#         """
#         os.makedirs(output_dir, exist_ok=True)
#         yolo_annotations = []
#         with open(annotation_file, "r") as file:
#             for line in file:
#                 x_min, y_min, x_max, y_max = map(int, line.split(",")[:4])  # Adjust parsing as per your data format
#                 yolo_annotations.append(
#                     self.convert_to_yolo(image_width, image_height, (x_min, y_min, x_max, y_max))
#                 )

#         output_file = os.path.join(output_dir, os.path.basename(annotation_file))
#         with open(output_file, "w") as file:
#             file.write("\n".join(yolo_annotations))

#     def batch_transform_to_yolo(self, annotation_dir, output_dir, image_size_dict):
#         """
#         Batch transform all annotation files in a directory to YOLO format.

#         Args:
#             annotation_dir (str): Directory containing annotation files in pixel format.
#             output_dir (str): Directory to save YOLO-format annotation files.
#             image_size_dict (dict): Dictionary mapping annotation filenames to (width, height) of images.
#         """
#         for annotation_file in os.listdir(annotation_dir):
#             if annotation_file.endswith(".txt"):
#                 image_width, image_height = image_size_dict[annotation_file]
#                 self.transform_annotations_to_yolo(
#                     os.path.join(annotation_dir, annotation_file),
#                     output_dir,
#                     image_width,
#                     image_height
#                 )
