import gdown
import os
import zipfile
from argparse import ArgumentParser

def download_traffic(destination):
        
    if not os.path.exists(destination):
        raise Exception(f"Destination {destination} doesn't exist")

    url = 'https://drive.google.com/uc?id=14J8A8bPoQFlhz7zYWErvFdPc0icTFvuF'
    filename = 'traffic.zip'
    file_path = os.path.join(destination, filename)
    print(f"Downloading {filename} from Google Drive...")
    gdown.download(url, file_path, quiet=False)

    print(f"Extracting {filename}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        extract_dir = os.path.join(destination, "traffic")
        zip_ref.extractall(extract_dir)
        print(f"{filename} extracted successfully.")

if __name__=="__main__":
    
    parser = ArgumentParser(description="Download traffic dataset to a specified destination.")
    parser.add_argument("-d", "--destination", type=str, required=True, help="Destination folder where the dataset will be downloaded.")
    args = parser.parse_args()
    download_traffic(args.destination)

    
    
