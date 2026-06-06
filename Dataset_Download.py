import pandas as pd
import kagglehub
import os
import urllib.request
import zipfile

class KaggleDatasetDownloader:
    def __init__(self, links, base_dir=os.path.join("Datasets", "Raw")):
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


def download():
    links = [
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        "rounakbanik/the-movies-dataset"]

    # Initialize and run Kaggle downloader
    downloader = KaggleDatasetDownloader(links)
    downloader.run()
    
def make_dataframe():
    folder_path=os.path.join("Datasets","Raw", "the-movies-dataset")
    files=['credits','keywords','links','links_small','movies_metadata','ratings_small','ratings']
    dataframes={}
    for file in files:
        file_path=os.path.join(folder_path, f"{file}.csv")
        dataframes[file] = pd.read_csv(file_path,low_memory=False)
    combined_links = pd.concat([dataframes['links'], dataframes['links_small']], axis=0, ignore_index=True).drop_duplicates()
    dataframes['movies_metadata']["id"] = pd.to_numeric(dataframes['movies_metadata']["id"], errors="coerce")
    combined_links["tmdbId"] = pd.to_numeric(
    combined_links["tmdbId"],
    errors="coerce"
)
    df = dataframes['credits'].merge(dataframes['keywords'],on="id", how="outer") \
        .merge(combined_links, left_on="id", right_on="tmdbId", how="outer") \
        .merge(dataframes['movies_metadata'],on="id", how="outer")
    combined_ratings = (
    pd.concat([dataframes['ratings'], dataframes['ratings_small']], ignore_index=True)
    .sort_values("timestamp")
    .drop_duplicates(subset=["userId", "movieId"], keep="last")
)
    return df, combined_ratings

if __name__ == "__main__":
    download()