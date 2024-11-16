"""
Feature Repository
==================

.. module:: pipeline.stages.feature_repo
   :synopsis: This module defines the entities, views, and services for the Feast feature store

.. moduleauthor:: aai540-group3
"""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Float32, Int64


# Define the primary entity for Feast
entities = Entity(
    name="patient",
    join_keys=["patient_id"],
)


def get_source(name: str) -> PostgreSQLSource:
    """Retrieve the PostgreSQL source for a given feature view.

    This function creates a PostgreSQLSource object configured to query all records from the specified feature table.

    :param name: The base name of the feature view to retrieve the source for.
    :type name: str
    :return: A PostgreSQLSource configured for the specified feature view.
    :rtype: PostgreSQLSource
    """
    return PostgreSQLSource(
        name=f"{name}_source",
        query="SELECT * FROM feast_diabetes_features",
        timestamp_field="event_timestamp",
    )


# Define Feature Views for various feature domains

admissions = FeatureView(
    name="admissions",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="admission_type_id_3", dtype=Int64),
        Field(name="admission_type_id_5", dtype=Int64),
        Field(name="admission_source_id_4", dtype=Int64),
        Field(name="admission_source_id_7", dtype=Int64),
        Field(name="admission_source_id_9", dtype=Int64),
        Field(name="discharge_disposition_id_2", dtype=Int64),
        Field(name="discharge_disposition_id_7", dtype=Int64),
        Field(name="discharge_disposition_id_10", dtype=Int64),
        Field(name="discharge_disposition_id_18", dtype=Int64),
    ],
    online=True,
    source=get_source("admissions"),
    tags={"domain": "admissions"},
)

demographic = FeatureView(
    name="demographic",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="age", dtype=Int64),
        Field(name="gender_1", dtype=Int64),
        Field(name="AfricanAmerican", dtype=Int64),
        Field(name="Asian", dtype=Int64),
        Field(name="Caucasian", dtype=Int64),
        Field(name="Hispanic", dtype=Int64),
        Field(name="Other", dtype=Int64),
    ],
    online=True,
    source=get_source("demographic"),
    tags={"domain": "demographic"},
)

clinical = FeatureView(
    name="clinical",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="time_in_hospital", dtype=Int64),
        Field(name="num_procedures", dtype=Int64),
        Field(name="num_medications", dtype=Int64),
        Field(name="number_diagnoses", dtype=Int64),
    ],
    online=True,
    source=get_source("clinical"),
    tags={"domain": "clinical"},
)

service = FeatureView(
    name="service",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="number_outpatient_log1p", dtype=Float32),
        Field(name="number_emergency_log1p", dtype=Float32),
        Field(name="number_inpatient_log1p", dtype=Float32),
    ],
    online=True,
    source=get_source("service"),
    tags={"domain": "service"},
)

labs = FeatureView(
    name="labs",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="max_glu_serum_0", dtype=Int64),
        Field(name="max_glu_serum_1", dtype=Int64),
        Field(name="A1Cresult_0", dtype=Int64),
        Field(name="A1Cresult_1", dtype=Int64),
    ],
    online=True,
    source=get_source("labs"),
    tags={"domain": "labs"},
)

medications = FeatureView(
    name="medications",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="metformin", dtype=Int64),
        Field(name="repaglinide", dtype=Int64),
        Field(name="nateglinide", dtype=Int64),
        Field(name="chlorpropamide", dtype=Int64),
        Field(name="glimepiride", dtype=Int64),
        Field(name="glipizide", dtype=Int64),
        Field(name="glyburide", dtype=Int64),
        Field(name="pioglitazone", dtype=Int64),
        Field(name="rosiglitazone", dtype=Int64),
        Field(name="acarbose", dtype=Int64),
        Field(name="tolazamide", dtype=Int64),
        Field(name="insulin", dtype=Int64),
        Field(name="glyburide-metformin", dtype=Int64),
    ],
    online=True,
    source=get_source("medications"),
    tags={"domain": "medications"},
)

diagnosis = FeatureView(
    name="diagnosis",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="level1_diag1_1.0", dtype=Float32),
        Field(name="level1_diag1_2.0", dtype=Float32),
        Field(name="level1_diag1_3.0", dtype=Float32),
        Field(name="level1_diag1_4.0", dtype=Float32),
        Field(name="level1_diag1_5.0", dtype=Float32),
        Field(name="level1_diag1_6.0", dtype=Float32),
        Field(name="level1_diag1_7.0", dtype=Float32),
        Field(name="level1_diag1_8.0", dtype=Float32),
    ],
    online=True,
    source=get_source("diagnosis"),
    tags={"domain": "diagnosis"},
)

interactions = FeatureView(
    name="interactions",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="num_medications|time_in_hospital", dtype=Float32),
        Field(name="num_medications|num_procedures", dtype=Float32),
        Field(name="time_in_hospital|num_lab_procedures", dtype=Float32),
        Field(name="num_medications|num_lab_procedures", dtype=Float32),
        Field(name="num_medications|number_diagnoses", dtype=Float32),
        Field(name="age|number_diagnoses", dtype=Float32),
        Field(name="change|num_medications", dtype=Float32),
        Field(name="number_diagnoses|time_in_hospital", dtype=Float32),
        Field(name="num_medications|numchange", dtype=Float32),
    ],
    online=True,
    source=get_source("interactions"),
    tags={"domain": "interactions"},
)

target = FeatureView(
    name="target",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="readmitted", dtype=Int64),
    ],
    online=True,
    source=get_source("target"),
    tags={"domain": "target"},
)

# Aggregate all feature views into a list for easy management
features = [
    demographic,
    clinical,
    service,
    labs,
    medications,
    diagnosis,
    interactions,
    admissions,
    target,
]

# Define a Feature Service that includes all feature views
feature_service = FeatureService(
    name="readmission_prediction",
    features=features,
)
