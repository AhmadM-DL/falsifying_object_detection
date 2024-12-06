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
        zip_ref.extractall(extract_dir)
        print(f"{train_filename} extracted successfully.")

if __name__=="__main__":
    
    parser = ArgumentParser(description="Download visdrone dataset to a specified destination.")
    parser.add_argument("-d", "--destination", type=str, required=True, help="Destination folder where the dataset will be downloaded.")
    args = parser.parse_args()
    
    
