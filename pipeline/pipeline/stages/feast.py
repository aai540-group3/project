import os
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import feast
import pandas as pd
import yaml
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.infra.utils.postgres.connection_utils import df_to_postgres_table
from feast.infra.utils.postgres.postgres_config import PostgreSQLConfig
from loguru import logger
from omegaconf import OmegaConf
from sqlalchemy import create_engine, text

from pipeline.stages.base import PipelineStage


class Feast(PipelineStage):
    """Pipeline stage for feature store setup using PostgreSQL."""

    @staticmethod
    def get_feast_dtype(series: pd.Series):
        """Get the Feast feature type based on the Pandas Series dtype."""
        if pd.api.types.is_integer_dtype(series):
            return feast.types.Int64
        elif pd.api.types.is_float_dtype(series):
            return feast.types.Float32
        elif pd.api.types.is_bool_dtype(series):
            return feast.types.Bool
        elif pd.api.types.is_string_dtype(series):
            return feast.types.String
        else:
            # Default to String if unknown type
            return feast.types.String

    @logger.catch(reraise=True)
    def run(self):
        """Set up and configure feature store with PostgreSQL."""
        logger.debug(f"Feast version: {feast.__version__}")

        # Set protocol buffers implementation
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        # Initialize feature repository path using absolute paths
        feature_repo_path = pathlib.Path(self.cfg.paths.feature_repo).resolve()
        feature_repo_path.mkdir(parents=True, exist_ok=True)

        # Set up source and destination paths for features file
        src_path = pathlib.Path(self.cfg.paths.processed) / "features_not_onehot.parquet"
        src_path = src_path.resolve()
        dest_path = feature_repo_path / "features_not_onehot.parquet"

        # Validate and copy features file
        if not src_path.exists():
            raise FileNotFoundError(f"Features file not found at {src_path}")
        shutil.copy2(src_path, dest_path)
        logger.info(f"COPIED: {src_path} --> {dest_path}")

        # Load and prepare data
        df = pd.read_parquet(dest_path)

        # Add required columns if missing
        if "event_timestamp" not in df.columns:
            df["event_timestamp"] = datetime.now(timezone.utc)
            logger.info("Added event_timestamp to features")

        if "patient_id" not in df.columns:
            df["patient_id"] = df.index
            logger.info("Added patient_id column")

        # Save modified DataFrame
        df.to_parquet(dest_path, index=False)
        logger.debug(f"Updated features saved to: {dest_path}")

        # Read Feast configuration from params.yaml using OmegaConf.to_container
        feast_config = OmegaConf.to_container(self.cfg.feast.config, resolve=True, enum_to_str=True)

        # Handle feature store configuration
        config_path = feature_repo_path / "feature_store.yaml"
        logger.debug(f"CREATING: {config_path}")

        # Ensure that 'config' key is not present
        if "config" in feast_config:
            feast_config = feast_config["config"]

        # Write the configuration to feature_store.yaml
        config_path.write_text(yaml.dump(feast_config, default_flow_style=False))
        logger.info(f"CREATED: {config_path}")

        # LOG CONFIG
        logger.debug(f"FEAST CONFIG:\n{config_path.read_text()}")

        # Initialize feature store with absolute path
        store = feast.FeatureStore(repo_path=str(feature_repo_path.absolute()))

        # Setup database and upload data
        self.setup_database(feast_config, df)
        self.upload_data_to_postgres(df, feast_config)

        # Define entity
        patient = feast.Entity(name="patient", join_keys=["patient_id"])

        # FEATURE GROUPS DEFINITION
        FEATURE_GROUPS = {
            "demographic": ["race", "gender", "age"],
            "clinical": [
                "time_in_hospital",
                "num_lab_procedures",
                "num_procedures",
                "num_medications",
                "number_diagnoses",
            ],
            "service": ["number_outpatient", "number_emergency", "number_inpatient"],
            "labs": ["max_glu_serum", "a1cresult"],
            "medications": [
                col for col in df.columns if any(med in col for med in ["metformin", "insulin", "glyburide"])
            ],
            "diagnosis": [col for col in df.columns if "diag" in col],
        }  # fmt: skip

        # DEFINE FEATURE VIEWS FOR EACH GROUP
        feature_views = []
        for domain, features in FEATURE_GROUPS.items():
            schema = []

            # Include the entity column 'patient_id' in the schema
            if "patient_id" in df.columns:
                dtype = self.get_feast_dtype(df["patient_id"])
                schema.append(feast.Field(name="patient_id", dtype=dtype))
            else:
                raise ValueError("The 'patient_id' column is missing from the DataFrame.")

            # Include the feature columns
            for feat in features:
                if feat in df.columns:
                    dtype = self.get_feast_dtype(df[feat])
                    schema.append(feast.Field(name=feat, dtype=dtype))
                else:
                    logger.warning(f"Feature '{feat}' not found in DataFrame columns.")

            fv = feast.FeatureView(
                name=domain,
                entities=[patient],
                ttl=timedelta(days=365),
                schema=schema,
                online=True,
                source=PostgreSQLSource(
                    name=f"{domain}_source",
                    query="SELECT * FROM feast_diabetes_features",
                    timestamp_field="event_timestamp",
                ),
                tags={"domain": domain},
            )
            feature_views.append(fv)

        # Define the feature service
        readmission_service = feast.FeatureService(
            name="readmission_prediction",
            features=feature_views,
        )

        # Apply all feature views and service to the store
        store.apply([patient, *feature_views, readmission_service])

        # Collect metrics for tracking
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domains": {
                domain: {"feature_count": len(features), "features": features}
                for domain, features in FEATURE_GROUPS.items()
            },
            "services": {
                "readmission_prediction": {
                    "feature_count": len(readmission_service.feature_view_projections),
                    "description": "Features for predicting readmission risk",
                }
            },
            "store_config": config_path.read_text(),
        }

        # Save metrics
        self.save_metrics("metrics", metrics)

        # Materialize features
        feature_view_names = [fv.name for fv in feature_views]
        end_date = datetime.now(timezone.utc)
        for feature_view_name in feature_view_names:
            store.materialize_incremental(end_date=end_date, feature_views=[feature_view_name])
            logger.info(f"Materialized feature view: {feature_view_name}")

        logger.info("Feature store setup completed with PostgreSQL storage")

    def setup_database(self, config, df: pd.DataFrame):
        """Initialize PostgreSQL database and required extensions."""
        # Database credentials
        postgres_host = config["online_store"]["host"]
        postgres_port = config["online_store"]["port"]
        postgres_database = config["online_store"]["database"]
        postgres_schema = config["online_store"]["db_schema"]
        postgres_user = config["online_store"]["user"]
        postgres_password = config["online_store"]["password"]

        # Create database if it doesn't exist
        init_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/postgres"
        default_engine = create_engine(init_url)

        with default_engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            result = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": postgres_database})

            if not result.scalar():
                logger.info(f"Creating database: {postgres_database}")
                conn.execute(text(f"CREATE DATABASE {postgres_database}"))

        default_engine.dispose()

        # Create schema if it doesn't exist
        db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_database}"
        engine = create_engine(db_url)

        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")

            # Create schema
            create_schema_sql = text(f"CREATE SCHEMA IF NOT EXISTS {postgres_schema}")
            conn.execute(create_schema_sql)

            # Generate CREATE TABLE SQL
            create_table_sql = self.generate_create_table_sql(df, postgres_schema)
            conn.execute(create_table_sql)

        # Dispose the engine when done
        engine.dispose()

    def generate_create_table_sql(self, df: pd.DataFrame, schema: str):
        """Generate the CREATE TABLE SQL statement based on the DataFrame columns and types."""
        # Map Pandas dtypes to PostgreSQL types
        dtype_mapping = {
            "int64": "BIGINT",
            "int32": "INTEGER",
            "float64": "DOUBLE PRECISION",
            "float32": "REAL",
            "bool": "BOOLEAN",
            "object": "TEXT",
            "datetime64[ns, UTC]": "TIMESTAMP WITH TIME ZONE",
            "datetime64[ns]": "TIMESTAMP",
        }

        columns_sql = []
        for column in df.columns:
            pandas_dtype = str(df[column].dtype)
            # Handle timezone-aware datetime
            if pandas_dtype.startswith("datetime64[ns"):
                if "tz=" in pandas_dtype:
                    sql_dtype = "TIMESTAMP WITH TIME ZONE"
                else:
                    sql_dtype = "TIMESTAMP"
            else:
                sql_dtype = dtype_mapping.get(pandas_dtype, "TEXT")

            if column == "event_timestamp":
                sql_dtype = "TIMESTAMP"  # Ensure event_timestamp is TIMESTAMP

            # Escape column names with double quotes
            columns_sql.append(f'    "{column}" {sql_dtype}')

        # Define primary key
        primary_key = 'PRIMARY KEY ("event_timestamp", "patient_id")'

        # Join columns with commas and newlines
        columns_definition = ",\n".join(columns_sql)

        create_table_sql_str = (
            f"CREATE TABLE IF NOT EXISTS {schema}.feast_diabetes_features ({columns_definition},\n{primary_key})"
        )

        return text(create_table_sql_str)

    def upload_data_to_postgres(self, df: pd.DataFrame, config):
        """Upload feature data to PostgreSQL."""
        postgres_host = config["online_store"]["host"]
        postgres_port = config["online_store"]["port"]
        postgres_database = config["online_store"]["database"]
        postgres_schema = config["online_store"]["db_schema"]
        postgres_user = config["online_store"]["user"]
        postgres_password = config["online_store"]["password"]

        # Create PostgreSQL connection configuration
        postgres_config = PostgreSQLConfig(
            host=postgres_host,
            port=int(postgres_port),
            database=postgres_database,
            db_schema=postgres_schema,
            user=postgres_user,
            password=postgres_password,
        )

        # Create connection string
        conn_string = (
            f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_database}"
        )
        engine = create_engine(conn_string)

        try:
            with engine.connect() as connection:
                # Drop the existing table if it exists
                connection.execute(text(f"DROP TABLE IF EXISTS {postgres_schema}.feast_diabetes_features"))
                connection.commit()

                logger.info("Dropped existing table")

            # Upload new data to Postgres
            df_to_postgres_table(
                config=postgres_config,
                df=df,
                table_name="feast_diabetes_features",
            )

            logger.info("Successfully uploaded data to Postgres")

        except Exception as e:
            logger.error(f"Error uploading data to Postgres: {str(e)}")
            raise
        finally:
            engine.dispose()
