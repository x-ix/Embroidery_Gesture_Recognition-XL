import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import os


def download_from_huggingface_dataset(repo_id: str, filename: str):
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    local_path = Path(filename)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(local_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Downloaded {filename}")
    return local_path

def unzip_and_delete(zip_path: Path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)
    print(f"Extracted and deleted {zip_path.name}")

def main():
    repo_id = "iix/Embroidery"
    files = [
        "collected_clips_(positive).zip",
        "collected_clips_(negative).zip"
    ]

    for file in files:
        folder = Path(file).stem
        if not Path(folder).exists():
            zip_path = download_from_huggingface_dataset(repo_id, file)
            unzip_and_delete(zip_path)
        else:
            print(f"Skipping {file}: {folder} already exists.")

if __name__ == "__main__":
    main()
