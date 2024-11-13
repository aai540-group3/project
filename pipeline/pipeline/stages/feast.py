import os
import pathlib
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import feast
import numpy as np
import pandas as pd
import psycopg
import pyarrow as pa
import yaml
from feast.type_map import arrow_to_pg_type
from loguru import logger
from omegaconf import OmegaConf
from psycopg.conninfo import make_conninfo
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from .feature_repo import entities, feature_service, features
from .stage import Stage


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL connection."""

    host: str
    port: int
    database: str
    user: str
    password: str
    db_schema: str = "public"
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


class Feast(Stage):
    """Pipeline stage for feature store setup with early hash checking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temp_dirs = []
        self._pg_manager = None

    def check_data_hash(self, src_path: pathlib.Path) -> bool:
        """Check if source data has changed by comparing hashes.
        Returns True if data has changed or hash doesn't exist."""
        try:
            # Read the source parquet file
            df = pd.read_parquet(src_path)

            # Calculate current data hash
            data_hash = pd.util.hash_pandas_object(df).sum()

            # Define hash file location
            hash_file = pathlib.Path(self.cfg.paths.feature_repo) / "data_hash.txt"

            # Check if hash file exists and compare hashes
            if hash_file.exists():
                with open(hash_file, "r") as f:
                    last_hash = int(f.read().strip())
                if last_hash == data_hash:
                    logger.info("Data hash matches previous run. Skipping feature store setup.")
                    return False

            # Save the new hash
            hash_file.parent.mkdir(parents=True, exist_ok=True)
            with open(hash_file, "w") as f:
                f.write(str(data_hash))
            logger.info("Data has changed or is new. Proceeding with feature store setup.")
            return True

        except Exception as e:
            logger.warning(f"Error checking data hash: {str(e)}. Will proceed with feature store setup.")
            return True

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
                with conn.cursor() as cur:
                    cur.execute(f'SET search_path TO "{self._pg_manager.config.db_schema}"')
                    cur.execute(verify_sql)
                    results = cur.fetchall()

                    for table_name, columns in results:
                        logger.info(f"Table {table_name} columns: {columns}")

            logger.info("Feature tables setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to set up feature tables: {str(e)}")
            # Log detailed error information
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            raise

    def _setup_feature_tables(self, df: pd.DataFrame):
        """Set up required feature tables in PostgreSQL aligned with feature views."""
        try:
            logger.info("Setting up feature tables...")

            # Ensure patient_id exists
            if "patient_id" not in df.columns:
                logger.info("Adding patient_id column")
                df["patient_id"] = range(1, len(df) + 1)

            # Ensure event_timestamp exists
            if "event_timestamp" not in df.columns:
                logger.info("Adding event_timestamp column")
                df["event_timestamp"] = pd.Timestamp(
                    self.cfg.dataset.get("timestamp", "1970-01-01T00:00:00Z")
                ).tz_convert("UTC")

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

            # Define required columns for each view
            required_columns = {
                "admissions": [
                    *[col for col in df.columns if col.startswith("admission_type_id_")],
                    *[col for col in df.columns if col.startswith("discharge_disposition_id_")],
                    *[col for col in df.columns if col.startswith("admission_source_id_")],
                ],
                "demographic": [
                    "age",
                    *[col for col in df.columns if col.startswith("race_")],
                ],
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
                    *[col for col in df.columns if col.startswith("number_outpatient_log1p")],
                    *[col for col in df.columns if col.startswith("number_emergency_log1p")],
                    *[col for col in df.columns if col.startswith("number_inpatient_log1p")],
                ],
                "labs": [
                    "max_glu_serum",
                    "A1Cresult",
                ],
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
                    *[col for col in df.columns if col.startswith("level1_diag1_")],
                ],
                "interactions": [
                    *[col for col in df.columns if "|" in col],
                ],
                "target": ["readmitted"],
            }

            # Add missing columns with default values
            for view_name, columns in required_columns.items():
                missing_cols = [col for col in columns if col not in df.columns]
                if missing_cols:
                    logger.info(f"Adding missing columns for {view_name} view: {missing_cols}")
                    for col in missing_cols:
                        # For specific clinical columns, use appropriate defaults
                        if col in ["num_lab_procedures", "num_medications", "number_diagnoses"]:
                            df[col] = df["num_procedures"] if "num_procedures" in df.columns else 0
                        else:
                            df[col] = 0
                    logger.info(f"Added {len(missing_cols)} missing columns with default values")

            # Update the main features table with the new columns (only once)
            self._pg_manager.create_table_from_df(
                df=df, table_name="feast_diabetes_features", if_exists="replace", schema="feast"
            )
            logger.info("Updated main features table with all required columns")

            # Verify table creation and column existence using cursor instead of pd.read_sql
            verify_sql = """
            SELECT 
                table_name,
                string_agg(column_name, ', ') as columns
            FROM information_schema.columns
            WHERE table_schema = 'feast'
            GROUP BY table_name;
            """

            with self._pg_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f'SET search_path TO "{self._pg_manager.config.db_schema}"')
                    cur.execute(verify_sql)
                    results = cur.fetchall()

                    for table_name, columns in results:
                        logger.info(f"Table {table_name} columns: {columns}")

                    # Add row count verification
                    for table_name in ["feast_diabetes_features", "feast_patient_entities"]:
                        cur.execute(f"SELECT COUNT(*) FROM feast.{table_name}")
                        count = cur.fetchone()[0]
                        logger.info(f"Table {table_name} contains {count} rows")

            logger.info("Feature tables setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to set up feature tables: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            raise

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

            # Check source data path exists
            src_path = pathlib.Path(self.cfg.paths.processed) / "features_not_onehot.parquet"
            if not src_path.exists():
                raise FileNotFoundError(f"Source data not found at: {src_path}")

            # If data hasn't changed, skip feature store setup
            if not self.check_data_hash(src_path):
                return

            online_store_cfg = feast_config.get("online_store", {})

            container = None
            try:
                # PostgreSQL container setup
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
                    .with_exposed_ports(online_store_cfg.get("port", 5432))
                    .with_volume_mapping(init_sql_path, "/docker-entrypoint-initdb.d/init.sql")
                    .with_bind_ports(online_store_cfg.get("port", 5432), online_store_cfg.get("port", 5432))
                )
                container.start()

                wait_for_logs(
                    container=container,
                    predicate="database system is ready to accept connections",
                    timeout=120,
                )
                logger.info("PostgreSQL container logs indicate database system is ready")

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

                # Feature repository setup
                feature_repo_path = pathlib.Path(self.cfg.paths.feature_repo).resolve()
                feature_repo_path.mkdir(parents=True, exist_ok=True)

                # Copy data for Feast
                src_path = pathlib.Path(self.cfg.paths.processed) / "features_not_onehot.parquet"
                dest_path = feature_repo_path / "features_not_onehot.parquet"
                shutil.copy2(src_path, dest_path)
                logger.info(f"Copied features file to: {dest_path}")

                # Set up feature tables in PostgreSQL
                df = self.load_data("features.parquet", "data/processed")
                self._setup_feature_tables(df)

                # Create feature store configuration
                with open(feature_repo_path / "feature_store.yaml", "w") as f:
                    yaml.dump(feast_config, f)
                logger.info(f"Created feature store config at: {feature_repo_path / 'feature_store.yaml'}")

                # Initialize and apply feature store
                store = feast.FeatureStore(repo_path=str(feature_repo_path))
                store.apply([entities, *features, feature_service])
                logger.info("Applied feature definitions.")

                # Materialize feature views
                try:
                    store.materialize_incremental(end_date=datetime.now(timezone.utc))
                    logger.info("Successfully materialized feature views.")
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
