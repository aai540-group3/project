import logging
import os
import boto3
from ucimlrepo import fetch_ucirepo
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("FETCHING: dataset")
        diabetes_data = fetch_ucirepo(id=296)
        X = diabetes_data.data.features
        y = diabetes_data.data.targets
        metadata = diabetes_data.metadata
        variables = diabetes_data.variables
        df = pd.concat([X, y], axis=1)

        logger.debug("PRINTING: metadata")
        logger.debug(json.dumps(metadata, indent=4))
        logger.debug("PRINTING: variables")
        variables_list = variables.to_dict(orient="records")
        logger.debug(json.dumps(variables_list, indent=4))
        logger.debug("HEAD: data")
        logger.debug(df.head())
        logger.debug("PRINTING: columns and data types")
        logger.debug(df.dtypes)

        os.makedirs("data/raw", exist_ok=True)

        logger.info("SAVING: data/raw/data.parquet")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, "data/raw/data.parquet")

        logger.info("SAVING: data/raw/data.csv")
        df.to_csv("data/raw/data.csv", index=False)

        logger.info("SAVING: data/raw/metadata.json")
        with open("data/raw/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info("SAVING: data/raw/variables.json")
        with open("data/raw/variables.json", "w") as f:
            json.dump(variables_list, f, indent=4)

        s3 = boto3.resource("s3", region_name="us-east-1")
        bucket = s3.Bucket("aai540-group3")

        logger.info("UPLOADING: data/raw/data.parquet")
        logger.info("ENDPOINT: s3://aai540-group3/datasets/data.parquet")
        bucket.upload_file("data/raw/data.parquet", "datasets/data.parquet")

        logger.info("UPLOADING: data/raw/data.csv")
        logger.info("ENDPOINT: s3://aai540-group3/datasets/data.csv")
        bucket.upload_file("data/raw/data.csv", "datasets/data.csv")

        logger.info("UPLOADING: data/raw/metadata.json")
        logger.info("ENDPOINT: s3://aai540-group3/datasets/metadata.json")
        bucket.upload_file("data/raw/metadata.json", "datasets/metadata.json")

        logger.info("UPLOADING: data/raw/variables.json")
        logger.info("ENDPOINT: s3://aai540-group3/datasets/variables.json")
        bucket.upload_file("data/raw/variables.json", "datasets/variables.json")

        logger.info("FINISHED: ingest")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
