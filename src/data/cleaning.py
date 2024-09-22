import os

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_paths = cfg.data.path

    raw_data_path = to_absolute_path(f"{data_paths.raw}/data.csv")
    interim_data_path = to_absolute_path(f"{data_paths.interim}/data.csv")

    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)

    # No cleaning steps implemented
    print("No cleaning steps. Passing data through.")

    # Save cleaned data
    os.makedirs(to_absolute_path(data_paths.interim), exist_ok=True)
    df.to_csv(interim_data_path, index=False)
    print(f"Interim data saved to {interim_data_path}")


if __name__ == "__main__":
    main()
