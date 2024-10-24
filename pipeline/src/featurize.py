import os

# Ensure compatibility with protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
from feast import (
    Entity,
    Feature,
    FeatureView,
    ValueType,
    FileSource,
    FeatureStore,
)
import numpy as np
import contextlib


@contextlib.contextmanager
def ignore_astimezone_error():
    """Temporarily suppresses errors from astimezone() calls."""
    try:
        with contextlib.suppress(TypeError):
            yield
    except Exception as e:
        # Log any other exceptions, but continue execution
        logging.error(f"An unexpected error occurred: {e}")


def get_feature_list(df, feature_columns):
    """Helper function to create feature list with explicit type checking"""
    feature_list = []
    for col in feature_columns:
        # Convert column name to string to ensure compatibility
        col = str(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            dtype = ValueType.DOUBLE
        else:
            dtype = ValueType.STRING
        feature_list.append(Feature(name=col, dtype=dtype))
    return feature_list


def map_diagnosis(code):
    """Maps diagnosis codes to disease categories"""
    if pd.isna(code):
        return "Unknown"
    if str(code).startswith("V") or str(code).startswith("E"):
        return "Other"
    try:
        code = float(code)
    except:
        return "Unknown"
    if (code >= 390 and code < 460) or (code == 785):
        return "Circulatory"
    elif (code >= 460 and code < 520) or (code == 786):
        return "Respiratory"
    elif (code >= 520 and code < 580) or (code == 787):
        return "Digestive"
    elif code >= 250 and code < 251:
        return "Diabetes"
    elif code >= 800 and code < 1000:
        return "Injury"
    elif code >= 710 and code < 740:
        return "Musculoskeletal"
    elif (code >= 580 and code < 630) or (code == 788):
        return "Genitourinary"
    elif code >= 140 and code < 240:
        return "Neoplasms"
    else:
        return "Other"


def process_features(df, logger):
    """Process and create features from the input dataframe"""
    # Store target variable
    y = df["readmitted"]
    df = df.drop("readmitted", axis=1)

    # Map diagnosis codes
    logger.info("Mapping diagnosis codes to disease categories...")
    diag_cols = ["diag_1", "diag_2", "diag_3"]
    for col in diag_cols:
        if col in df.columns:
            df[col] = df[col].apply(map_diagnosis)
        else:
            logger.warning(f"Diagnosis column '{col}' not found in the data.")

    # Create dummy variables for diagnosis categories
    df_diag = pd.get_dummies(df[diag_cols], prefix=diag_cols)
    df = pd.concat([df.drop(diag_cols, axis=1), df_diag], axis=1)

    # Process medication features
    logger.info("Processing medication features...")
    medication_cols = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]

    for col in medication_cols:
        if col in df.columns:
            df[col] = df[col].replace({"Up": 1, "Down": 1, "Steady": 1, "No": 0})
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["num_medications_prescribed"] = df[medication_cols].sum(axis=1)

    # Process lab results
    logger.info("Processing lab result features...")
    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}

    if "max_glu_serum" in df.columns:
        df["max_glu_serum"] = df["max_glu_serum"].replace(glu_map)
    if "a1cresult" in df.columns:
        df["a1cresult"] = df["a1cresult"].replace(a1c_map)

    # Process discharge disposition
    logger.info("Processing discharge disposition...")
    discharge_map = {
        1: "Home",
        2: "Short-term Hospital",
        3: "SNF",
        4: "ICF",
        5: "Another Type of Facility",
        6: "Home Health Service",
        7: "AMA",
        8: "Home Health Service",
        9: "Unknown/Invalid",
        10: "Expired",
        11: "Hospice/Home",
        12: "Hospice/Medical Facility",
        13: "Another Hospital",
    }

    if "discharge_disposition_id" in df.columns:
        df["discharge_disposition"] = (
            df["discharge_disposition_id"]
            .astype(int)
            .map(lambda x: discharge_map.get(x, "Other"))
        )
        df = df.drop("discharge_disposition_id", axis=1)
        df = pd.get_dummies(df, columns=["discharge_disposition"], prefix="discharge")

    # Process admission features
    logger.info("Processing admission features...")
    admission_type_map = {
        1: "Emergency",
        2: "Urgent",
        3: "Elective",
        4: "Newborn",
        5: "Not Available",
        6: "NULL",
        7: "Trauma Center",
        8: "Not Mapped",
        9: "Unknown/Invalid",
    }

    admission_source_map = {
        1: "Physician Referral",
        2: "Clinic Referral",
        3: "HMO Referral",
        4: "Transfer from Hospital",
        5: "Transfer from SNF",
        6: "Transfer from Another Health Care Facility",
        7: "Emergency Room",
        8: "Court/Law Enforcement",
        9: "Not Available",
        10: "Transfer from Crit Care",
        11: "Normal Delivery",
        12: "Premature Delivery",
        13: "Sick Baby",
        14: "Extramural Birth",
        15: "Not Available",
        17: "NULL",
        18: "Transfer from Another Home Health Agency",
        19: "Readmission to Same Home Health Agency",
        20: "Not Mapped",
        21: "Unknown/Invalid",
    }

    if "admission_type_id" in df.columns:
        df["admission_type"] = (
            df["admission_type_id"]
            .astype(int)
            .map(lambda x: admission_type_map.get(x, "Other"))
        )
        df = df.drop("admission_type_id", axis=1)
        df = pd.get_dummies(df, columns=["admission_type"], prefix="admission_type")

    if "admission_source_id" in df.columns:
        df["admission_source"] = (
            df["admission_source_id"]
            .astype(int)
            .map(lambda x: admission_source_map.get(x, "Other"))
        )
        df = df.drop("admission_source_id", axis=1)
        df = pd.get_dummies(df, columns=["admission_source"], prefix="admission_source")

    # Create interaction features
    logger.info("Creating interaction features...")
    df["num_medications_time_in_hospital"] = (
        df["num_medications"] * df["time_in_hospital"]
    )
    df["procedures_per_diagnosis"] = df["num_procedures"] / (df["number_diagnoses"] + 1)
    df["medications_per_procedure"] = df["num_medications"] / (df["num_procedures"] + 1)
    df["diagnoses_per_visit"] = df["number_diagnoses"] / df["time_in_hospital"]

    # Handle infinite values and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Drop less useful columns
    columns_to_drop = ["change", "diabetesmed"]
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)

    # Reattach target
    df["readmitted"] = y.reset_index(drop=True)

    return df


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    feast_logger = logging.getLogger("feast")
    feast_logger.setLevel(logging.DEBUG)

    # Define paths
    interim_data_path = "data/interim/data_cleaned.parquet"
    featured_data_path = "data/interim/data_featured.parquet"

    try:
        logger.info("Loading preprocessed data...")
        df = pd.read_parquet(interim_data_path)
        if "id" not in df.columns:
            logger.error("'id' column not found in the data.")
            return
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        logger.error(f"Data file not found: {interim_data_path}")
        return

    # Process features
    df = process_features(df, logger)

    # Add timestamps for Feast
    logger.info("Adding timestamp fields...")
    start_date = datetime(1999, 1, 1, tzinfo=timezone.utc)
    df["event_timestamp"] = pd.date_range(
        start=start_date, periods=len(df), freq="T", tz=timezone.utc
    )
    df["created_timestamp"] = datetime.now(timezone.utc)

    # Save processed data
    logger.info(f"Saving featured data to {featured_data_path}...")
    os.makedirs(os.path.dirname(featured_data_path), exist_ok=True)
    df.to_parquet(featured_data_path, index=False)

    # Set up Feast
    logger.info("Setting up Feast feature store...")
    record = Entity(name="id", value_type=ValueType.INT64, description="Patient ID")

    absolute_path = os.path.abspath(featured_data_path)
    data_source = FileSource(
        path=absolute_path,
        event_timestamp_column="event_timestamp",
        created_timestamp_column="created_timestamp",
    )

    # Define features
    feature_columns = list(df.columns)
    exclude_columns = ["readmitted", "event_timestamp", "created_timestamp", "id"]
    feature_columns = [col for col in feature_columns if col not in exclude_columns]

    # Create feature list
    feature_list = get_feature_list(df, feature_columns)

    # Create feature view
    diabetes_feature_view = FeatureView(
        name="diabetes_features",
        entities=[record.name],
        ttl=timedelta(days=365 * 10),
        features=list(feature_list),
        batch_source=data_source,
        online=True,
    )

    # Initialize and apply feature store
    store = FeatureStore(repo_path=".")
    store.apply([record, diabetes_feature_view])

    # Materialize features for specific dates
    logger.info("Materializing features for specific dates...")
    try:
        dates_to_fetch = df["event_timestamp"].dt.date.unique()

        for target_date in dates_to_fetch:
            start_date = pd.to_datetime(target_date)

            # Ensure start_date is timezone-aware (UTC)
            if start_date.tz is None:
                start_date = start_date.tz_localize(timezone.utc)
            else:
                start_date = start_date.tz_convert(timezone.utc)

            end_date = start_date + pd.Timedelta(days=1)

            # Apply the context manager here
            with ignore_astimezone_error():
                store.materialize(start_date=start_date, end_date=end_date)
                store.apply([record, diabetes_feature_view])

            logger.info(f"Features materialized for {target_date} successfully")

    except Exception as e:
        logger.error(f"Materialization error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
