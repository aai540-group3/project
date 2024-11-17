"""
Postgres and Feast Management
============================

.. module:: pipeline.stages.postgres_feast
   :synopsis: This module manages PostgreSQL operations and the Feast feature store

.. moduleauthor:: aai540-group3
"""

import os
import pathlib
import shutil
import tempfile
import urllib.parse
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
from sqlalchemy import create_engine
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from .feature_repo import entities, feature_service, features
from .stage import Stage


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL connection.

    :param host: The hostname of the PostgreSQL server.
    :type host: str
    :param port: The port number on which PostgreSQL is running.
    :type port: int
    :param database: The name of the PostgreSQL database.
    :type database: str
    :param user: The username for PostgreSQL authentication.
    :type user: str
    :param password: The password for PostgreSQL authentication.
    :type password: str
    :param db_schema: The schema within the PostgreSQL database, defaults to "public".
    :type db_schema: str, optional
    :param min_conn: The minimum number of connections in the pool, defaults to 1.
    :type min_conn: int, optional
    :param max_conn: The maximum number of connections in the pool, defaults to 10.
    :type max_conn: int, optional
    :param keepalives_idle: The number of seconds of inactivity after which a keepalive message is sent, defaults to 30.
    :type keepalives_idle: int, optional
    :param sslmode: The SSL mode for the PostgreSQL connection, defaults to None.
    :type sslmode: str, optional
    :param sslkey_path: Path to the SSL key file, defaults to None.
    :type sslkey_path: str, optional
    :param sslcert_path: Path to the SSL certificate file, defaults to None.
    :type sslcert_path: str, optional
    :param sslrootcert_path: Path to the SSL root certificate file, defaults to None.
    :type sslrootcert_path: str, optional
    """

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
    """Wait for PostgreSQL to be ready for connections.

    :param host: The hostname of the PostgreSQL server.
    :type host: str
    :param port: The port number on which PostgreSQL is running.
    :type port: int
    :param user: The username for PostgreSQL authentication.
    :type user: str
    :param password: The password for PostgreSQL authentication.
    :type password: str
    :param database: The name of the PostgreSQL database.
    :type database: str
    :param max_retries: Maximum number of retry attempts, defaults to 30.
    :type max_retries: int, optional
    :param retry_interval: Time in seconds between retries, defaults to 2.
    :type retry_interval: int, optional
    :return: `True` if PostgreSQL becomes ready within the retry limit.
    :rtype: bool
    :raises OperationalError: If PostgreSQL does not become ready within the retry limit.
    """
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
    """Manager class for PostgreSQL operations.

    This class handles connections, schema management, table creation from DataFrames,
    query execution, and connection pool management for PostgreSQL.
    """

    def __init__(self, config: PostgresConfig):
        """Initialize with PostgreSQL configuration.

        :param config: The configuration object containing PostgreSQL connection details.
        :type config: PostgresConfig
        """
        self.config = config
        self._pool = None

    def _get_conninfo(self) -> str:
        """Generate connection info string.

        :return: The connection info string for psycopg.
        :rtype: str
        """
        return make_conninfo(
            conninfo="",
            user=self.config.user,
            password=self.config.password,
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.database,
        )

    def get_connection(self) -> psycopg.Connection:
        """Get a single database connection.

        :return: A psycopg connection object.
        :rtype: psycopg.Connection
        """
        return psycopg.connect(
            conninfo=self._get_conninfo(),
            keepalives_idle=self.config.keepalives_idle,
        )

    def ensure_schema_exists(self):
        """Ensure the specified schema exists with retry logic.

        :raises Exception: If the schema cannot be created after all retries.
        """
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
        """Create a table from DataFrame and return schema.

        :param df: The pandas DataFrame to create a table from.
        :type df: pd.DataFrame
        :param table_name: The name of the table to create.
        :type table_name: str
        :param if_exists: Behavior when the table already exists ('fail', 'replace', 'ignore'), defaults to "fail".
        :type if_exists: str, optional
        :param schema: The schema under which to create the table, defaults to None.
        :type schema: str, optional
        :return: A dictionary mapping column names to their numpy data types.
        :rtype: Dict[str, np.dtype]
        :raises Exception: If table creation or data insertion fails.
        """
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
        """Generate SQL for table creation from DataFrame.

        :param df: The pandas DataFrame to generate SQL from.
        :type df: pd.DataFrame
        :param table_name: The name of the table to create.
        :type table_name: str
        :param schema: The schema under which to create the table, defaults to None.
        :type schema: str, optional
        :return: The SQL statement for creating the table.
        :rtype: str
        """
        target_schema = schema or self.config.db_schema
        pa_table = pa.Table.from_pandas(df)
        columns = [f""""{f.name}" {arrow_to_pg_type(str(f.type))}""" for f in pa_table.schema]
        return f"""
            CREATE TABLE IF NOT EXISTS "{target_schema}"."{table_name}" (
                {", ".join(columns)}
            );
        """

    def execute_query(self, query: str, params: Optional[List[Any]] = None):
        """Execute a query with optional parameters.

        :param query: The SQL query to execute.
        :type query: str
        :param params: A list of parameters to bind to the query, defaults to None.
        :type params: List[Any], optional
        :raises Exception: If query execution fails.
        """
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
    """Pipeline stage for feature store setup using PostgreSQL with pgvector.

    This stage handles the setup and configuration of the Feast feature store,
    including PostgreSQL management, pgvector extension setup, feature table creation,
    and feature repository configuration.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Feast stage with necessary parameters.

        :param args: Variable length argument list.
        :type args: Any
        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: Any
        """
        super().__init__(*args, **kwargs)
        self._temp_dirs = []
        self._pg_manager = None

    def _setup_postgres_manager(self, db_config: Dict[str, Any]) -> PostgresManager:
        """Set up PostgreSQL manager with configuration.

        :param db_config: A dictionary containing PostgreSQL connection details.
        :type db_config: Dict[str, Any]
        :return: An instance of PostgresManager initialized with the provided configuration.
        :rtype: PostgresManager
        """
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
        """Set up pgvector extension.

        :param pg_manager: An instance of PostgresManager to execute queries.
        :type pg_manager: PostgresManager
        :raises Exception: If pgvector extension setup fails.
        """
        try:
            pg_manager.execute_query("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("Successfully created pgvector extension")
        except Exception as e:
            logger.error(f"Failed to create pgvector extension: {str(e)}")
            raise

    def verify_tables(self):
        """Verify table creation and column existence using SQLAlchemy.

        :return: A pandas DataFrame containing table names and their columns.
        :rtype: pd.DataFrame
        :raises Exception: If table verification fails.
        """
        # Create SQLAlchemy engine
        encoded_password = urllib.parse.quote_plus(self.cfg.feast.config.online_store.password)
        engine_url = (
            f"postgresql://{self.cfg.feast.config.online_store.user}:"
            f"{encoded_password}@"
            f"{self.cfg.feast.config.online_store.host}:"
            f"{self.cfg.feast.config.online_store.port}/"
            f"{self.cfg.feast.config.online_store.database}"
        )
        engine = create_engine(engine_url)

        verify_sql = """
        SELECT 
            table_name,
            string_agg(column_name, ', ') as columns
        FROM information_schema.columns
        WHERE table_schema = 'feast'
        GROUP BY table_name;
        """

        try:
            # Use SQLAlchemy engine with pandas
            tables_df = pd.read_sql(verify_sql, engine)

            # Log table information
            for _, row in tables_df.iterrows():
                logger.info(f"Table {row['table_name']} columns: {row['columns']}")

            return tables_df

        except Exception as e:
            logger.error(f"Failed to verify tables: {str(e)}")
            raise
        finally:
            engine.dispose()

    def _setup_feature_tables(self, df: pd.DataFrame):
        """Set up required feature tables in PostgreSQL aligned with feature views.

        :param df: The pandas DataFrame containing feature data.
        :type df: pd.DataFrame
        :raises Exception: If feature table setup fails.
        """
        try:
            logger.info("Setting up feature tables...")

            # Create the main features table that all feature views source from
            self._pg_manager.create_table_from_df(
                df=df,
                table_name="feast_diabetes_features",
                if_exists="replace",
                schema="feast",
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
                df=entity_df,
                table_name="feast_patient_entities",
                if_exists="replace",
                schema="feast",
            )
            logger.info("Created entity reference table: feast_patient_entities")

            self.verify_tables()

        except Exception as e:
            logger.error(f"Failed to set up feature tables: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            raise

    def data_has_changed(self, df: pd.DataFrame) -> bool:
        """Check if data has changed based on a hash.

        If the hash file is corrupt or invalid, it will be recreated.

        :param df: The pandas DataFrame to check for changes.
        :type df: pd.DataFrame
        :return: `True` if data has changed or hash is invalid/missing, `False` otherwise.
        :rtype: bool
        """
        data_hash = pd.util.hash_pandas_object(df).sum()
        hash_file = pathlib.Path(self.cfg.paths.feature_repo) / "data_hash.txt"

        def write_hash(hash_value: int):
            """Helper function to write hash value to file.

            :param hash_value: The hash value to write.
            :type hash_value: int
            """
            hash_file.parent.mkdir(parents=True, exist_ok=True)
            with open(hash_file, "w") as f:
                f.write(str(hash_value))

        def read_hash() -> int:
            """Helper function to read the hash value from the file.

            :return: The hash value read from the file.
            :rtype: int
            :raises ValueError: If the hash file is empty.
            :raises OSError: If the hash file cannot be read.
            """
            with open(hash_file, "r") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Hash file is empty.")
                return int(content)

        try:
            if hash_file.exists():
                try:
                    last_hash = read_hash()
                    if last_hash == data_hash:
                        logger.info("Data has not changed. Skipping preparation.")
                        return False
                except (ValueError, OSError) as e:
                    logger.warning(f"Corrupt or invalid hash file detected ({e}). Recreating hash file.")

            # Either hash file doesn't exist, data has changed, or hash was invalid
            write_hash(data_hash)
            return True

        except Exception as e:
            logger.error(f"Unexpected error handling hash file: {str(e)}. Proceeding with data preparation.")
            # In case of any other errors, assume data has changed
            return True

    def _prepare_feature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature data by handling missing values and ensuring correct data types.

        :param df: The pandas DataFrame to prepare.
        :type df: pd.DataFrame
        :return: The prepared pandas DataFrame.
        :rtype: pd.DataFrame
        :raises Exception: If data preparation fails.
        """
        if not self.data_has_changed(df):
            return df

        try:
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

    def is_docker_running(self):
        """Check if Docker daemon is running.

        :return: `True` if Docker is running, `False` otherwise.
        :rtype: bool
        """
        try:
            import docker

            client = docker.from_env()
            client.ping()  # Ping the Docker daemon to check if it's responsive
            print("Docker is running.")
            return True
        except docker.DockerException as e:
            print("Docker is not running or is inaccessible:", str(e))
            return False

    def run(self):
        """Set up and configure feature store with PostgreSQL and pgvector extension.

        This method performs the following steps:
            1. Checks if Docker is running.
            2. Loads and validates Feast configuration.
            3. Starts a PostgreSQL container with pgvector extension.
            4. Waits for PostgreSQL to be ready.
            5. Initializes PostgreSQL manager.
            6. Copies feature data and configurations.
            7. Prepares feature data.
            8. Sets up feature tables in PostgreSQL.
            9. Initializes and applies Feast feature store.
           10. Saves exploration metrics and cleans up.

        :raises RuntimeError: If Docker is not running.
        :raises ValueError: If Feast configuration is missing required fields.
        :raises Exception: If any step in the setup process fails.
        """
        try:
            logger.debug(f"Feast version: {feast.__version__}")

            # Check if Docker is running before continuing
            if not self.is_docker_running():
                raise RuntimeError("Docker is not running. Please start Docker and try again.")

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
                    .with_env(
                        "POSTGRES_USER",
                        online_store_cfg.get("user", "postgres"),
                    )
                    .with_env(
                        "POSTGRES_PASSWORD",
                        online_store_cfg.get("password", "postgres"),
                    )
                    .with_env(
                        "POSTGRES_DB",
                        online_store_cfg.get("database", "registry"),
                    )
                    .with_exposed_ports(online_store_cfg.get("port", 5432))
                    .with_volume_mapping(
                        init_sql_path,
                        "/docker-entrypoint-initdb.d/init.sql",
                    )
                    .with_bind_ports(
                        online_store_cfg.get("port", 5432),
                        online_store_cfg.get("port", 5432),
                    )
                )
                container.start()

                # Wait for database to be ready with better logging
                wait_for_logs(
                    container=container,
                    predicate="database system is ready to accept connections",
                    timeout=120,
                )
                logger.info("PostgreSQL container logs indicate database system is ready")

                # Additional wait for init process
                wait_for_logs(
                    container=container,
                    predicate="PostgreSQL init process complete",
                    timeout=120,
                )
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
                required_fields = [
                    "project",
                    "provider",
                    "entity_key_serialization_version",
                    "coerce_tz_aware",
                ]
                top_level_config = {
                    field: feast_config.pop(field, None) for field in required_fields if field in feast_config
                }
                feast_config = {**top_level_config, **feast_config}

                # Feature repository setup
                feature_repo_path = pathlib.Path(self.cfg.paths.feature_repo).resolve()
                feature_repo_path.mkdir(parents=True, exist_ok=True)

                # Copy feature data
                src_path = pathlib.Path(self.cfg.paths.processed) / "features.parquet"
                dest_path = feature_repo_path / "features.parquet"
                shutil.copy2(src_path, dest_path)
                logger.info(f"Copied features file to: {dest_path}")

                # Create feature store configuration
                with open(feature_repo_path / "feature_store.yaml", "w") as f:
                    yaml.dump(feast_config, f)
                logger.info(f"Created feature store config at: {feature_repo_path / 'feature_store.yaml'}")

                # Copy feature definitions
                feature_repo_source = pathlib.Path(__file__).parent / "feature_repo.py"
                shutil.copy2(feature_repo_source, feature_repo_path / "feature_repo.py")
                logger.info(f"Copied feature definitions to: {feature_repo_path / 'feature_repo.py'}")

                # Load and prepare feature data
                df = pd.read_parquet(dest_path)

                # Add event_timestamp if it doesn't exist
                if "event_timestamp" not in df.columns:
                    # Retrieve fixed timestamp from configuration
                    fixed_timestamp = datetime.fromisoformat(self.cfg.dataset.timestamp)

                    # Validate fixed_timestamp is a datetime object
                    if not isinstance(fixed_timestamp, datetime):
                        raise TypeError("self.cfg.dataset.fixed_timestamp must be a datetime object")

                    # Ensure fixed_timestamp is timezone-aware
                    if fixed_timestamp.tzinfo is None:
                        logger.debug("Localizing fixed_timestamp to UTC")
                        fixed_timestamp = fixed_timestamp.replace(tzinfo=timezone.utc)
                    logger.debug("Adding 'event_timestamp' column with fixed_timestamp")
                    df["event_timestamp"] = pd.Timestamp(fixed_timestamp).round("s")
                    logger.info("Created 'event_timestamp' column with fixed_timestamp.")

                elif not pd.api.types.is_datetime64_any_dtype(df["event_timestamp"]):
                    logger.debug("Converting 'event_timestamp' to datetime")
                    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])

                # Ensure 'event_timestamp' is timezone-aware
                if df["event_timestamp"].dt.tz is None:
                    logger.debug("Making 'event_timestamp' timezone-aware with UTC")
                    df["event_timestamp"] = df["event_timestamp"].dt.tz_localize("UTC")

                # Create a unique patient ID if it doesn't exist
                if "patient_id" not in df.columns:
                    df["patient_id"] = range(1, len(df) + 1)
                    logger.info("Created 'patient_id' column.")
                elif not df["patient_id"].is_unique:
                    logger.warning("'patient_id' column is not unique. Overwriting with unique IDs.")
                    df["patient_id"] = range(1, len(df) + 1)

                # Prepare feature data
                df = self._prepare_feature_data(df)

                # Set up feature tables in PostgreSQL
                self._setup_feature_tables(df)

                # Save prepared data back to parquet
                df.to_parquet(dest_path, index=False)

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

                if container and self.cfg.feast.stop_container:
                    container.stop()
                    logger.info("Stopped PostgreSQL container.")

        finally:
            for temp_dir in self._temp_dirs:
                shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary directories.")
