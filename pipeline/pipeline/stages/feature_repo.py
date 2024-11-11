from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Float32, Int64

entities = Entity(
    name="patient",
    join_keys=["patient_id"],
)


def get_source(name):
    return PostgreSQLSource(
        name=f"{name}_source", query="SELECT * FROM feast_diabetes_features", timestamp_field="event_timestamp"
    )


admissions = FeatureView(
    name="admissions",
    entities=[entities],
    ttl=timedelta(days=365),
    schema=[
        Field(name="patient_id", dtype=Int64),
        Field(name="admission_type_id_3", dtype=Int64),
        Field(name="admission_type_id_4", dtype=Int64),
        Field(name="admission_type_id_5", dtype=Int64),
        Field(name="discharge_disposition_id_2.0", dtype=Float32),
        Field(name="discharge_disposition_id_3.5", dtype=Float32),
        Field(name="admission_source_id_4.0", dtype=Float32),
        Field(name="admission_source_id_9.0", dtype=Float32),
        Field(name="admission_source_id_11.0", dtype=Float32),
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
        Field(name="race_Asian", dtype=Int64),
        Field(name="race_Caucasian", dtype=Int64),
        Field(name="race_Hispanic", dtype=Int64),
        Field(name="race_Other", dtype=Int64),
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
        Field(name="num_lab_procedures", dtype=Int64),
        Field(name="num_procedures", dtype=Int64),
        Field(name="num_medications", dtype=Int64),
        Field(name="number_diagnoses", dtype=Int64),
        Field(name="num_med", dtype=Int64),
        Field(name="num_change", dtype=Int64),
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
        Field(name="service_utilization", dtype=Float32),
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
        Field(name="max_glu_serum", dtype=Int64),
        Field(name="A1Cresult", dtype=Int64),
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
        Field(name="acetohexamide", dtype=Int64),
        Field(name="glipizide", dtype=Int64),
        Field(name="glyburide", dtype=Int64),
        Field(name="tolbutamide", dtype=Int64),
        Field(name="pioglitazone", dtype=Int64),
        Field(name="rosiglitazone", dtype=Int64),
        Field(name="acarbose", dtype=Int64),
        Field(name="miglitol", dtype=Int64),
        Field(name="troglitazone", dtype=Int64),
        Field(name="tolazamide", dtype=Int64),
        Field(name="insulin", dtype=Int64),
        Field(name="glyburide-metformin", dtype=Int64),
        Field(name="glipizide-metformin", dtype=Int64),
        Field(name="glimepiride-pioglitazone", dtype=Int64),
        Field(name="metformin-rosiglitazone", dtype=Int64),
        Field(name="metformin-pioglitazone", dtype=Int64),
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
        Field(name="level1_diag1_1", dtype=Int64),
        Field(name="level1_diag1_2", dtype=Int64),
        Field(name="level1_diag1_3", dtype=Int64),
        Field(name="level1_diag1_4", dtype=Int64),
        Field(name="level1_diag1_5", dtype=Int64),
        Field(name="level1_diag1_6", dtype=Int64),
        Field(name="level1_diag1_7", dtype=Int64),
        Field(name="level1_diag1_8", dtype=Int64),
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
        Field(name="num_med|time_in_hospital", dtype=Float32),
        Field(name="num_med|num_procedures", dtype=Float32),
        Field(name="time_in_hospital|num_lab_procedures", dtype=Float32),
        Field(name="num_med|num_lab_procedures", dtype=Float32),
        Field(name="num_med|number_diagnoses", dtype=Float32),
        Field(name="age|number_diagnoses", dtype=Float32),
        Field(name="change|num_med", dtype=Float32),
        Field(name="number_diagnoses|time_in_hospital", dtype=Float32),
        Field(name="num_med|num_change", dtype=Float32),
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

feature_service = FeatureService(
    name="readmission_prediction",
    features=features,
)
