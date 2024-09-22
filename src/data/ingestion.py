import os

import datasets
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_paths = cfg.data.path
    dataset_name = cfg.data.dataset_name
    data_output_path = to_absolute_path(data_paths.raw)

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset(dataset_name, token=None)
    df = dataset["train"].to_pandas()

    # Create the output directory if it doesn't exist
    os.makedirs(data_output_path, exist_ok=True)

    # Save to data/raw/
    print("Saving raw data...")
    df.to_csv(f"{data_output_path}/data.csv", index=False)
    print(f"Data saved to {data_output_path}/data.csv")


if __name__ == "__main__":
    main()
