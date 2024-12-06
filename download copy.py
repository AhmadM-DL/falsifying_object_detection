import gdown
import os
import zipfile
from convert_visdrone_to_yolo import conv_visdrone_2_yolo
from argparse import ArgumentParser

def download_visdrone(destination):
        
    if not os.path.exists(destination):
        raise Exception("Destination doesn't exist")

    train_url = 'https://drive.google.com/uc?id=1iabGGSMVMQufGWzA8o3rFoBYXQLVTMFv'
    train_filename = 'visdrone_train.zip'
    file_path = os.path.join(destination, train_filename)
    print(f"Downloading {train_filename} from Google Drive...")
    gdown.download(train_url, file_path, quiet=False)

    print(f"Extracting {train_filename}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        extract_dir = os.path.join(destination, "visdrone_train")

    # Prepare paths for YOLO conversion
    source_annot_dir = os.path.join(extract_dir, "VisDrone2019-DET-train","annotations")  # Path to annotations
    source_image_dir = os.path.join(extract_dir, "VisDrone2019-DET-train","images")       # Path to images
    target_annot_dir = os.path.join(extract_dir,"VisDrone2019-DET-train","yolo_annotations")  # Path for YOLO annotations

    # Create the target annotation directory if it doesn't exist
    os.makedirs(target_annot_dir, exist_ok=True)

    # Debugging: Verify the contents of the directories
    print("Checking contents of annotations and images directories:")


    # Convert annotations to YOLO format
    print(f"Converting VisDrone annotations to YOLO format for {train_filename}...")
    try:
        conv_visdrone_2_yolo(
            source_annot_dir=source_annot_dir,
            source_image_dir=source_image_dir,
            target_annot_dir=target_annot_dir,
            low_dim_cutoff=400,   # Example cutoff for bounding box dimensions
            low_area_cutoff=0.01  # Example cutoff for bounding box area
        )
        print(f"Conversion completed for {train_filename}.")
    except Exception as e:
        print(f"Error during conversion for {train_filename}: {e}")

    print("Download, extraction, and annotation conversion completed!")

if __name__=="__main__":
    
    parser = ArgumentParser(description="Download visdrone dataset to a specified destination.")
    parser.add_argument("-d", "--destination", type=str, required=True, help="Destination folder where the dataset will be downloaded.")
    args = parser.parse_args()
    download_visdrone(args.destination)
    
    
