import os
import pathlib
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import feast
import pandas as pd
import yaml
import numpy as np
import psycopg
import pyarrow as pa
from psycopg.conninfo import make_conninfo
from loguru import logger
from omegaconf import OmegaConf
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from feast.type_map import arrow_to_pg_type

from pipeline.stages.base import PipelineStage
from pipeline.stages.feature_repo import entities, feature_service, features


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL connection."""

    host: str
    port: int
    database: str
    user: str
    password: str
    db_schema: str = "public"  # Default schema
    min_conn: int = 1
    max_conn: int = 10
    keepalives_idle: int = 30
    sslmode: Optional[str] = None
    sslkey_path: Optional[str] = None
    sslcert_path: Optional[str] = None
    sslrootcert_path: Optional[str] = None


def wait_for_postgres(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    max_retries: int = 30,
    retry_interval: int = 2,
):
    """Wait for PostgreSQL to be ready for connections."""
    import time
    from psycopg import OperationalError

    for attempt in range(max_retries):
        try:
            conn = psycopg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=database,
            )
            conn.close()
            logger.info("Successfully connected to PostgreSQL")
            return True
        except OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Database not ready yet: {str(e)}")
                time.sleep(retry_interval)
            else:
                logger.error("Max retries reached waiting for database")
                raise
    return False


class PostgresManager:
    """Manager class for PostgreSQL operations."""

    def __init__(self, config: PostgresConfig):
        """Initialize with PostgreSQL configuration."""
        self.config = config
        self._pool = None

    def _get_conninfo(self) -> str:
        """Generate connection info string."""
        return make_conninfo(
            conninfo="",
            user=self.config.user,
            password=self.config.password,
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.database,
        )

    def get_connection(self) -> psycopg.Connection:
        """Get a single database connection."""
        return psycopg.connect(
            conninfo=self._get_conninfo(),
            keepalives_idle=self.config.keepalives_idle,
        )

    def ensure_schema_exists(self):
        """Ensure the specified schema exists with retry logic."""
        import time

        max_retries = 5
        retry_interval = 2

        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.config.db_schema}"')
                        cur.execute(f'SET search_path TO "{self.config.db_schema}"')
                        conn.commit()
                        logger.info(f"Ensured schema exists and set search path to: {self.config.db_schema}")
                        return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} to create schema failed: {str(e)}")
                    time.sleep(retry_interval)
                else:
                    logger.error("Failed to create schema after all retries")
                    raise

    def create_table_from_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "fail",
        schema: Optional[str] = None,
    ) -> Dict[str, np.dtype]:
        """Create a table from DataFrame and return schema."""
        target_schema = schema or self.config.db_schema
        self.ensure_schema_exists()

        nr_columns = df.shape[1]
        placeholders = ", ".join(["%s"] * nr_columns)
        values = df.replace({np.NaN: None}).to_numpy().tolist()

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f'SET search_path TO "{target_schema}"')

                if if_exists == "replace":
                    cur.execute(f'DROP TABLE IF EXISTS "{target_schema}"."{table_name}"')

                create_sql = self._generate_create_table_sql(df, table_name, schema=target_schema)
                logger.debug(f"Creating table with SQL: {create_sql}")
                cur.execute(create_sql)

                if if_exists != "ignore":
                    insert_sql = f'INSERT INTO "{target_schema}"."{table_name}" VALUES ({placeholders})'
                    cur.executemany(insert_sql, values)

                conn.commit()
                logger.info(f"Successfully created and populated table: {target_schema}.{table_name}")

        return dict(zip(df.columns, df.dtypes))

    def _generate_create_table_sql(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: Optional[str] = None,
    ) -> str:
        """Generate SQL for table creation from DataFrame."""
        target_schema = schema or self.config.db_schema
        pa_table = pa.Table.from_pandas(df)
        columns = [f""""{f.name}" {arrow_to_pg_type(str(f.type))}""" for f in pa_table.schema]
        return f"""
            CREATE TABLE IF NOT EXISTS "{target_schema}"."{table_name}" (
                {", ".join(columns)}
            );
        """

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> None:
        """Execute a query with optional parameters."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f'SET search_path TO "{self.config.db_schema}"')
                cur.execute(query, params)
                conn.commit()

    def close(self):
        """Close connection pool if it exists."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None


class Feast(PipelineStage):
    """Pipeline stage for feature store setup using PostgreSQL with pgvector."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temp_dirs = []
        self._pg_manager = None

    def _setup_postgres_manager(self, db_config: Dict[str, Any]) -> PostgresManager:
        """Set up PostgreSQL manager with configuration."""
        config = PostgresConfig(
            host=db_config["host"],
            port=int(db_config["port"]),
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            db_schema=db_config.get("db_schema", "feature_store"),
        )
        return PostgresManager(config)

    def _setup_pgvector(self, pg_manager: PostgresManager):
        """Set up pgvector extension."""
        try:
            pg_manager.execute_query("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("Successfully created pgvector extension")
        except Exception as e:
            logger.error(f"Failed to create pgvector extension: {str(e)}")
            raise

    def _setup_feature_tables(self, df: pd.DataFrame):
        """Set up required feature tables in PostgreSQL aligned with feature views."""
        try:
            logger.info("Setting up feature tables...")

            # Create the main features table that all feature views source from
            self._pg_manager.create_table_from_df(
                df=df, table_name="feast_diabetes_features", if_exists="replace", schema="feast"
            )
            logger.info("Created main features table: feast_diabetes_features")

            # Create index on the main table
            index_sql = """
            CREATE INDEX IF NOT EXISTS idx_diabetes_features_patient 
            ON feast.feast_diabetes_features(patient_id, event_timestamp);
            """
            self._pg_manager.execute_query(index_sql)
            logger.info("Created index on feast_diabetes_features")

            # Create patient entity reference table
            entity_cols = ["patient_id", "event_timestamp"]
            entity_df = df[entity_cols].copy().drop_duplicates()
            self._pg_manager.create_table_from_df(
                df=entity_df, table_name="feast_patient_entities", if_exists="replace", schema="feast"
            )
            logger.info("Created entity reference table: feast_patient_entities")

            # Verify all required columns for feature views exist
            required_columns = {
                "admissions": [
                    "admission_type_id_3",
                    "admission_type_id_4",
                    "admission_type_id_5",
                    "discharge_disposition_id_2.0",
                    "discharge_disposition_id_3.5",
                    "admission_source_id_4.0",
                    "admission_source_id_9.0",
                    "admission_source_id_11.0",
                ],
                "demographic": ["age", "race_Asian", "race_Caucasian", "race_Hispanic", "race_Other"],
                "clinical": [
                    "time_in_hospital",
                    "num_lab_procedures",
                    "num_procedures",
                    "num_medications",
                    "number_diagnoses",
                    "num_med",
                    "num_change",
                ],
                "service": [
                    "service_utilization",
                    "number_outpatient_log1p",
                    "number_emergency_log1p",
                    "number_inpatient_log1p",
                ],
                "labs": ["max_glu_serum", "A1Cresult"],
                "medications": [
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
                    "insulin",
                    "glyburide-metformin",
                    "glipizide-metformin",
                    "glimepiride-pioglitazone",
                    "metformin-rosiglitazone",
                    "metformin-pioglitazone",
                ],
                "diagnosis": [
                    "level1_diag1_1",
                    "level1_diag1_2",
                    "level1_diag1_3",
                    "level1_diag1_4",
                    "level1_diag1_5",
                    "level1_diag1_6",
                    "level1_diag1_7",
                    "level1_diag1_8",
                ],
                "interactions": [
                    "num_med|time_in_hospital",
                    "num_med|num_procedures",
                    "time_in_hospital|num_lab_procedures",
                    "num_med|num_lab_procedures",
                    "num_med|number_diagnoses",
                    "age|number_diagnoses",
                    "change|num_med",
                    "number_diagnoses|time_in_hospital",
                    "num_med|num_change",
                ],
                "target": ["readmitted"],
            }

            # Check for missing columns and log warnings
            for view_name, columns in required_columns.items():
                missing_cols = [col for col in columns if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns for {view_name} view: {missing_cols}")

            # Add any missing binary columns with default values
            for view_name, columns in required_columns.items():
                for col in columns:
                    if col not in df.columns:
                        logger.warning(f"Adding missing column {col} with default value 0")
                        df[col] = 0

            # Update the main features table with any new columns
            self._pg_manager.create_table_from_df(
                df=df, table_name="feast_diabetes_features", if_exists="replace", schema="feast"
            )
            logger.info("Updated main features table with all required columns")

            # Verify table creation and column existence
            verify_sql = """
            SELECT 
                table_name,
                string_agg(column_name, ', ') as columns
            FROM information_schema.columns
            WHERE table_schema = 'feast'
            GROUP BY table_name;
            """
            with self._pg_manager.get_connection() as conn:
                tables_df = pd.read_sql(verify_sql, conn)

            for _, row in tables_df.iterrows():
                logger.info(f"Table {row['table_name']} columns: {row['columns']}")

        except Exception as e:
            logger.error(f"Failed to set up feature tables: {str(e)}")
            # Log detailed error information
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            raise

    def _prepare_feature_data(self, src_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature data for Feast ingestion."""
        logger.info("Started preparing feature data")
        df = src_df.copy()

        try:
            # Add/update event_timestamp
            if "event_timestamp" not in df.columns:
                logger.debug("Adding event_timestamp column")
                df["event_timestamp"] = datetime.now(timezone.utc)
            elif not pd.api.types.is_datetime64_any_dtype(df["event_timestamp"]):
                logger.debug("Converting event_timestamp to datetime")
                df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])

            # Ensure event_timestamp is timezone-aware
            if df["event_timestamp"].dt.tz is None:
                logger.debug("Making event_timestamp timezone-aware")
                df["event_timestamp"] = df["event_timestamp"].dt.tz_localize("UTC")

            # Add/update patient_id if needed
            if "patient_id" not in df.columns:
                logger.debug("Adding patient_id column from index")
                df["patient_id"] = df.index

            # Convert patient_id to string if it's not already
            df["patient_id"] = df["patient_id"].astype(str)

            # Handle null values
            numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns

            # Fill numeric nulls with median
            for col in numeric_columns:
                if df[col].isnull().any():
                    logger.debug(f"Filling null values in {col} with median")
                    df[col] = df[col].fillna(df[col].median())

            # Fill categorical nulls with mode
            for col in categorical_columns:
                if df[col].isnull().any():
                    logger.debug(f"Filling null values in {col} with mode")
                    df[col] = df[col].fillna(df[col].mode()[0])

            # Ensure all required columns have correct types
            type_conversions = {
                "age": "float64",
                "gender": "float64",
                "race": "str",
                "time_in_hospital": "float64",
                "num_lab_procedures": "float64",
                "num_procedures": "float64",
                "num_medications": "float64",
                "number_diagnoses": "float64",
                "max_glu_serum": "float64",
                "A1Cresult": "float64",
                "metformin": "float64",
                "insulin": "float64",
            }

            for col, dtype in type_conversions.items():
                if col in df.columns:
                    logger.debug(f"Converting {col} to {dtype}")
                    try:
                        if dtype == "str":
                            df[col] = df[col].astype(str)
                        else:
                            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")

            # Convert boolean columns to float64 (Feast requirement)
            bool_columns = df.select_dtypes(include=["bool"]).columns
            for col in bool_columns:
                logger.debug(f"Converting boolean column {col} to float64")
                df[col] = df[col].astype(float)

            # Create entity columns if needed for feature views
            entity_time_cols = ["patient_id", "event_timestamp"]
            missing_cols = [col for col in entity_time_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Final validation
            required_cols = ["patient_id", "event_timestamp"]
            for col in required_cols:
                if df[col].isnull().any():
                    raise ValueError(f"Column {col} contains null values")

            # Add created_timestamp if needed
            if "created_timestamp" not in df.columns:
                logger.debug("Adding created_timestamp column")
                df["created_timestamp"] = datetime.now(timezone.utc)

            logger.info(f"Feature data preparation completed. Shape: {df.shape}")
            logger.debug(f"Columns in prepared data: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Error during feature data preparation: {str(e)}")
            raise

    @logger.catch()
    def run(self):
        """Set up and configure feature store with PostgreSQL and pgvector extension."""
        try:
            logger.debug(f"Feast version: {feast.__version__}")

            # Load and validate feast configuration
            feast_config = OmegaConf.to_container(self.cfg.feast.config, resolve=True)
            if not feast_config:
                raise ValueError("Feast configuration is missing.")
            for key in ["project", "provider", "offline_store", "online_store"]:
                if key not in feast_config:
                    raise ValueError(f"Missing required Feast configuration: {key}")

            online_store_cfg = feast_config.get("online_store", {})

            container = None
            try:
                # Create init.sql file for pgvector extension
                init_sql_content = """
                CREATE SCHEMA IF NOT EXISTS feast;
                CREATE EXTENSION IF NOT EXISTS vector;
                """
                temp_dir = tempfile.mkdtemp()
                self._temp_dirs.append(temp_dir)
                init_sql_path = os.path.join(temp_dir, "init.sql")

                with open(init_sql_path, "w") as f:
                    f.write(init_sql_content)

                container = (
                    DockerContainer("pgvector/pgvector:pg16")
                    .with_env("POSTGRES_USER", online_store_cfg.get("user", "postgres"))
                    .with_env("POSTGRES_PASSWORD", online_store_cfg.get("password", "postgres"))
                    .with_env("POSTGRES_DB", online_store_cfg.get("database", "registry"))
                    .with_exposed_ports(5432)
                    .with_volume_mapping(init_sql_path, "/docker-entrypoint-initdb.d/init.sql")
                    .with_bind_ports(5432, 5432)
                )
                container.start()

                # Wait for database to be ready with better logging
                log_string_to_wait_for = "database system is ready to accept connections"
                wait_for_logs(container=container, predicate=log_string_to_wait_for, timeout=120)
                logger.info("PostgreSQL container logs indicate database system is ready")

                # Additional wait for init process
                init_log_string_to_wait_for = "PostgreSQL init process complete"
                wait_for_logs(container=container, predicate=init_log_string_to_wait_for, timeout=120)
                logger.info("PostgreSQL init process complete according to logs")

                exposed_port = container.get_exposed_port(5432)
                logger.info(f"Started pgvector/pgvector:pg16 container on port: {exposed_port}")

                # Configure database connection
                db_config = {
                    "host": "localhost",
                    "type": "postgres",
                    "user": online_store_cfg.get("user", "postgres"),
                    "password": online_store_cfg.get("password", "postgres"),
                    "database": online_store_cfg.get("database", "registry"),
                    "port": container.get_exposed_port(5432),
                    "db_schema": "feast",
                }

                # Wait for PostgreSQL to be fully ready
                wait_for_postgres(
                    host=db_config["host"],
                    port=db_config["port"],
                    user=db_config["user"],
                    password=db_config["password"],
                    database=db_config["database"],
                )

                # Initialize PostgreSQL manager
                self._pg_manager = self._setup_postgres_manager(db_config)
                self._pg_manager.ensure_schema_exists()
                logger.info("PostgreSQL manager initialized with schema")

                # Update store configurations
                for store in ["offline_store", "online_store"]:
                    if store in feast_config:
                        feast_config[store].update(db_config)

                if "online_store" in feast_config:
                    feast_config["online_store"].update(
                        {
                            "type": "postgres",
                            "vector_enabled": True,
                            "vector_len": 2,
                        }
                    )

                # Ensure required top-level fields
                required_fields = ["project", "provider", "entity_key_serialization_version", "coerce_tz_aware"]
                top_level_config = {
                    field: feast_config.pop(field, None) for field in required_fields if field in feast_config
                }
                feast_config = {**top_level_config, **feast_config}

                # Feature repository setup
                feature_repo_path = pathlib.Path(self.cfg.paths.feature_repo).resolve()
                feature_repo_path.mkdir(parents=True, exist_ok=True)

                # Process feature data
                src_path = pathlib.Path(self.cfg.paths.processed) / "features_not_onehot.parquet"
                dest_path = feature_repo_path / "features_not_onehot.parquet"
                shutil.copy2(src_path, dest_path)
                logger.info(f"Copied features file to: {dest_path}")

                # Load and prepare feature data
                df = pd.read_parquet(dest_path)
                df = self._prepare_feature_data(df)

                # Set up feature tables in PostgreSQL
                self._setup_feature_tables(df)

                # Save prepared data back to parquet
                df.to_parquet(dest_path, index=False)

                # Create feature store configuration
                with open(feature_repo_path / "feature_store.yaml", "w") as f:
                    yaml.dump(feast_config, f)
                logger.info(f"Created feature store config at: {feature_repo_path / 'feature_store.yaml'}")

                # Copy feature definitions
                feature_repo_source = pathlib.Path(__file__).parent / "feature_repo.py"
                shutil.copy2(feature_repo_source, feature_repo_path / "feature_repo.py")
                logger.info(f"Copied feature definitions to: {feature_repo_path / 'feature_repo.py'}")

                # Initialize and apply feature store
                store = feast.FeatureStore(repo_path=str(feature_repo_path))
                store.apply([entities, *features, feature_service])
                logger.info("Applied feature definitions.")

                # Try to materialize with better error handling
                try:
                    store.materialize_incremental(end_date=datetime.now(timezone.utc))
                    logger.info("Successfully materialized feature views.")
                except Exception as e:
                    logger.error(f"Failed to materialize feature views: {str(e)}")
                    # Log additional diagnostic information
                    logger.debug("Checking table existence and permissions...")
                    try:
                        self._pg_manager.execute_query(
                            "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = 'feast';"
                        )
                    except Exception as check_e:
                        logger.error(f"Failed to check tables: {str(check_e)}")
                    raise e
                    # Materialize feature views
                    try:
                        store.materialize_incremental(end_date=datetime.now(timezone.utc))
                        logger.info("Materialized feature views.")
                    except Exception as e:
                        logger.error(f"Failed to materialize feature views: {str(e)}")
                        raise

                    # Save metrics
                    metrics = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "feast_version": feast.__version__,
                        "feature_views": [fv.name for fv in store.list_feature_views()],
                        "feature_services": [fs.name for fs in store.list_feature_services()],
                        "entities": [e.name for e in store.list_entities()],
                    }
                    self.save_metrics("metrics", metrics)
                    logger.info("Feature store setup completed.")

            finally:
                if self._pg_manager:
                    self._pg_manager.close()
                    logger.info("Closed PostgreSQL manager.")

                if container:
                    container.stop()
                    logger.info("Stopped PostgreSQL container.")

        finally:
            for temp_dir in self._temp_dirs:
                shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary directories.")
