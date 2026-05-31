import pandas as pd
import kagglehub
import os
import urllib.request
import zipfile

class KaggleDatasetDownloader:
    def __init__(self, links, base_dir="Datasets"):
        self.links = links
        self.base_dir = os.path.abspath(base_dir)

    def run(self):
        os.makedirs(self.base_dir, exist_ok=True)

        for dataset in self.links:
            dataset_name = dataset.split('/')[-1]
            save_dir = os.path.join(self.base_dir, dataset_name)

            #  Skip if directory exists and has .csv files 
            if os.path.exists(save_dir) and any(f.endswith(".csv") for f in os.listdir(save_dir)):
                continue

            print(f"\nDownloading: {dataset}")
            path = kagglehub.dataset_download(dataset)
            
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving to: {save_dir}")

            for file in os.listdir(path):
                if file.endswith(".csv"):
                    print(f"  → {file}")
                    df = pd.read_csv(os.path.join(path, file), low_memory=False)
                    df.to_csv(os.path.join(save_dir, file), index=False)

        print("\nAll Kaggle datasets processed!")

    def add_dataset(self, dataset_link):
        self.links.append(dataset_link)


class MovieLensDownloader:
    def __init__(self):
        self.url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        self.base_save_dir = os.path.abspath("Datasets")
        self.dataset_name = "movielens-25m"

        self.save_dir = os.path.join(self.base_save_dir, self.dataset_name)
        self.zip_path = os.path.join(self.save_dir, "ml-25m.zip")

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)

        #  Skip if already extracted 
        if any(f.endswith(".csv") for f in os.listdir(self.save_dir)):
            return

        #  Skip download if zip already exists 
        if os.path.exists(self.zip_path):
            pass
        else:
            print(f"\nDownloading {self.dataset_name}. Please wait, this might take a moment...")
            urllib.request.urlretrieve(self.url, self.zip_path)
            print("Download complete!")

        # Extract
        print("Extracting CSV files...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)

                if filename:
                    print(f"Extracting: {filename}")

                    source = zip_ref.open(member)
                    target_path = os.path.join(self.save_dir, filename)

                    with open(target_path, "wb") as target:
                        target.write(source.read())

        print("Done!")


def download():
    links = [
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        "rounakbanik/the-movies-dataset"
    ]

    # Initialize and run Kaggle downloader
    downloader = KaggleDatasetDownloader(links)
    downloader.run()

    # Initialize and run MovieLens downloader
    ml_downloader = MovieLensDownloader()
    ml_downloader.run()


if __name__ == "__main__":
    download()