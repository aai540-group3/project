import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting data preprocessing pipeline")

    # Define data paths
    raw_data_path = "data/raw/data.csv"
    interim_data_path = "data/interim/data_cleaned.parquet"

    try:
        # Read CSV data with proper options
        logger.info(f"Reading raw data from {raw_data_path}")
        df = pd.read_csv(raw_data_path, low_memory=False)
    except FileNotFoundError as e:
        logger.error(f"Error: The file at {raw_data_path} was not found.")
        logger.error(e)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {raw_data_path}")
        logger.error(e)
        sys.exit(1)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Standardize 'gender' values
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.capitalize()

    # Generate a unique 'id' column based on the DataFrame's index
    df.reset_index(inplace=True)
    df.rename(columns={"index": "id"}, inplace=True)
    logger.info("Generated unique 'id' column for each record")

    # Remove duplicates
    logger.info("Removing duplicate entries")
    df.drop_duplicates(inplace=True)

    # Handle missing values for specific columns
    logger.info("Handling missing values")

    # Replace missing '?' with NaN
    df.replace("?", pd.NA, inplace=True)

    # Drop columns with too many missing values
    columns_to_drop = ["weight", "payer_code", "medical_specialty"]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Impute missing values in 'race' with the mode
    if "race" in df.columns:
        df["race"].fillna(df["race"].mode()[0], inplace=True)

    # Impute missing values in diagnosis columns with a placeholder
    diagnosis_columns = ["diag_1", "diag_2", "diag_3"]
    for col in diagnosis_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Remove entries with invalid 'gender' values
    if "gender" in df.columns:
        df = df[df["gender"].isin(["Male", "Female"])]

    # Map 'age' to numerical values (midpoint of age ranges)
    if "age" in df.columns:
        logger.info("Encoding 'age' as numerical values")
        age_map = {
            "[0-10)": 5,
            "[10-20)": 15,
            "[20-30)": 25,
            "[30-40)": 35,
            "[40-50)": 45,
            "[50-60)": 55,
            "[60-70)": 65,
            "[70-80)": 75,
            "[80-90)": 85,
            "[90-100)": 95,
        }
        df["age"] = df["age"].map(age_map)

    # Convert 'readmitted' to binary outcome
    logger.info("Converting 'readmitted' to binary outcome")
    df["readmitted"] = df["readmitted"].replace({"NO": 0, ">30": 0, "<30": 1})

    # Standardize medication column names
    medication_columns = [
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
    medication_columns = [col.lower() for col in medication_columns]

    # Ensure medication columns are in the dataframe
    available_medication_columns = [
        col for col in medication_columns if col in df.columns
    ]

    # Define the target variable
    target = "readmitted"

    # Separate features and target
    logger.info("Separating features and target variable")
    cols_to_drop = [target]
    X = df.drop(columns=cols_to_drop)
    y = df[target]

    # Identify numeric and categorical columns
    logger.info("Identifying numeric and categorical columns")
    # Exclude diagnosis, medication, and 'id' columns from categorical features
    all_categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    exclude_columns = diagnosis_columns + available_medication_columns + ["id"]
    categorical_features = [
        col for col in all_categorical_features if col not in exclude_columns
    ]
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Remove 'id' from numeric features
    numeric_features = [col for col in numeric_features if col != "id"]

    logger.debug(f"Numeric Features: {numeric_features}")
    logger.debug(
        f"Categorical Features (excluding diagnosis, medication, and 'id' columns): {categorical_features}"
    )
    logger.debug(f"Diagnosis Columns: {diagnosis_columns}")
    logger.debug(f"Medication Columns: {available_medication_columns}")
    logger.debug(f"'id' Column: ['id']")

    # Define preprocessing steps for numeric features
    logger.info("Setting up preprocessing pipelines")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Define preprocessing steps for categorical features
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # Combine preprocessing for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",  # Keep diagnosis, medication, and 'id' columns as is
    )

    # Create a preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Fit and transform the data
    logger.info("Starting data transformation")
    try:
        X_processed = preprocessing_pipeline.fit_transform(X)
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        return

    # Get feature names after one-hot encoding
    logger.info("Processing feature names and creating final DataFrame")
    try:
        onehot_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = onehot_encoder.get_feature_names_out(
            categorical_features
        ).tolist()

        # Get passthrough feature names (diagnosis, medication, and 'id' columns)
        passthrough_indices = preprocessor.transformers_[-1][2]
        passthrough_features = [X.columns[i] for i in passthrough_indices]

        # Combine all feature names
        feature_names = numeric_features + cat_feature_names + passthrough_features

        # Convert the processed features into a DataFrame
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        # Ensure correct data types for passthrough features
        X_df[passthrough_features] = X_df[passthrough_features].astype(object)

        # Display the preprocessed data overview
        logger.debug("Preprocessed Data Overview:")
        logger.debug(f"Shape: {X_df.shape}")
        logger.debug(f"Feature Names: {feature_names}")
        logger.debug("First Few Rows:")
        logger.debug(X_df.head())

        # Combine features and target for saving
        final_df = X_df.copy()
        final_df["readmitted"] = y.reset_index(drop=True)

        # Create output directory if it doesn't exist
        os.makedirs("data/interim", exist_ok=True)

        logger.info(f"Saving processed data to {interim_data_path}")
        final_df.to_parquet(interim_data_path, index=False)
        logger.info(f"Successfully saved processed data to {interim_data_path}")

    except Exception as e:
        logger.error(
            f"An error occurred while processing feature names or saving data: {e}"
        )
        return

    logger.info("Data preprocessing pipeline completed successfully")


if __name__ == "__main__":
    main()
