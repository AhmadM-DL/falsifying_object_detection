import gdown
import os
import zipfile

# Ensure the directory exists
directory = "C:/Users/noname/Desktop/me/aub/5.fall_24-25/eece_490/project/dataset/"
os.makedirs(directory, exist_ok=True)

# URLs with the file IDs
file_urls = [
    ('https://drive.google.com/uc?id=1iabGGSMVMQufGWzA8o3rFoBYXQLVTMFv', 'visdrone_train.zip'),
    ('https://drive.google.com/uc?id=1iXBWbKVP5B7_X5QFDUasXu0U3weBIRpZ', 'visdrone_val.zip'),
    ('https://drive.google.com/uc?id=1icc3r88LQGP-h6PSr9BYAfxlAVPdnGcO', 'visdrone_test.zip')
]

# Download and extract each file
for url, filename in file_urls:
    # Download the zip file
    file_path = os.path.join(directory, filename)
    gdown.download(url, file_path, quiet=False)

    # Extract the downloaded zip file
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(directory)  # Extract to the specified directory
        print(f"{filename} extracted successfully.")

print("Download and extraction completed!")
