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
        "rounakbanik/the-movies-dataset"]

    # Initialize and run Kaggle downloader
    downloader = KaggleDatasetDownloader(links)
    downloader.run()

    # Initialize and run MovieLens downloader
    ml_downloader = MovieLensDownloader()
    ml_downloader.run()
def make_dataframe():
    folder_path=os.path.join("Datasets", "the-movies-dataset")
    files=['credits','keywords','links','links_small','movies_metadata','ratings_small','ratings']
    dataframes={}
    for file in files:
        file_path=os.path.join(folder_path, f"{file}.csv")
        dataframes[file] = pd.read_csv(file_path,low_memory=False)
    credits_df=dataframes['credits']
    keywords_df=dataframes['keywords']
    links_df=dataframes['links']
    links_small_df=dataframes['links_small']
    movies_metadata_df=dataframes['movies_metadata']
    ratings_df=dataframes['ratings']
    ratings_small_df=dataframes['ratings_small']
    combined_links = pd.concat([links_df, links_small_df], axis=0, ignore_index=True).drop_duplicates()
    movies_metadata_df["id"] = pd.to_numeric(movies_metadata_df["id"], errors="coerce")
    combined_links["tmdbId"] = pd.to_numeric(
    combined_links["tmdbId"],
    errors="coerce"
)
    df = credits_df.merge(keywords_df,on="id", how="outer") \
        .merge(combined_links, left_on="id", right_on="tmdbId", how="outer") \
        .merge(movies_metadata_df,on="id", how="outer")
    combined_ratings = (
    pd.concat([ratings_df, ratings_small_df], ignore_index=True)
    .sort_values("timestamp")
    .drop_duplicates(subset=["userId", "movieId"], keep="last")
)
    return df, combined_ratings

if __name__ == "__main__":
    download()