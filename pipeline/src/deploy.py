import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import ModelCard, ModelCardData

# Setup logging and environment
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = Path("huggingface/models/diabetes-readmission")
REPO_ID = "aai540-group3/diabetes-readmission"
MODEL_DIRS = {
    "logistic_regression": {
        "root": Path("models/logistic_regression/artifacts"),
        "model": Path("models/logistic_regression/artifacts/model/model.joblib"),
        "metrics": Path("models/logistic_regression/artifacts/metrics/metrics.json"),
        "feature_importance": Path(
            "models/logistic_regression/artifacts/metrics/feature_importance.csv"
        ),
        "plots": Path("models/logistic_regression/artifacts/plots"),
        "scaler": Path("models/logistic_regression/artifacts/model/scaler.joblib"),
    },
    "neural_network": {
        "root": Path("models/neural_network/artifacts"),
        "model": Path("models/neural_network/artifacts/model/model.keras"),
        "metrics": Path("models/neural_network/artifacts/metrics/metrics.json"),
        "plots": Path("models/neural_network/artifacts/plots"),
        "scaler": Path("models/neural_network/artifacts/model/scaler.joblib"),
    },
    "autogluon": {
        "root": Path("models/autogluon/artifacts"),
        "model": Path("models/autogluon/artifacts/model/predictor.pkl"),
        "metrics": Path("models/autogluon/artifacts/metrics/metrics.json"),
        "feature_importance": Path(
            "models/autogluon/artifacts/model/feature_importance.csv"
        ),
        "plots": Path("models/autogluon/artifacts/plots"),
    },
}


def get_hf_token() -> str:
    """Get Hugging Face token from environment."""
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN environment variable must be set")
        raise EnvironmentError("HF_TOKEN environment variable must be set")
    return token


def load_metrics(metrics_path: Path) -> Optional[Dict]:
    """Load metrics from a JSON file."""
    try:
        if not metrics_path.exists():
            logger.error(f"Metrics file not found: {metrics_path}")
            return None

        with metrics_path.open() as f:
            metrics = json.load(f)

        required_metrics = {"test_accuracy", "test_auc"}
        if not all(metric in metrics for metric in required_metrics):
            logger.error(f"Missing required metrics in {metrics_path}")
            return None

        return metrics
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metrics JSON from {metrics_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading metrics from {metrics_path}: {e}")
        return None


def create_preprocessing_config(output_dir: Path, model_type: str) -> None:
    """Create preprocessing configuration for feature handling."""
    config = {
        "preprocessor": {
            "numeric_features": [
                "age",
                "time_in_hospital",
                "num_lab_procedures",
                "num_procedures",
                "num_medications",
                "number_diagnoses",
                "total_medications",
                "medication_density",
                "numchange",
                "nummed",
                "total_encounters",
                "encounter_per_time",
                "procedures_per_day",
                "lab_procedures_per_day",
                "procedures_to_medications",
                "diagnoses_per_encounter",
                "number_outpatient_log1p",
                "number_emergency_log1p",
                "number_inpatient_log1p",
            ],
            "binary_features": ["gender", "diabetesmed", "change", "insulin_with_oral"],
            "medication_features": [
                "metformin",
                "repaglinide",
                "nateglinide",
                "chlorpropamide",
                "glimepiride",
                "glipizide",
                "glyburide",
                "pioglitazone",
                "rosiglitazone",
                "acarbose",
                "miglitol",
                "insulin",
                "glyburide-metformin",
                "tolazamide",
                "metformin-pioglitazone",
                "metformin-rosiglitazone",
                "glimepiride-pioglitazone",
                "glipizide-metformin",
                "troglitazone",
                "tolbutamide",
                "acetohexamide",
            ],
            "interaction_features": [
                "num_medications_x_time_in_hospital",
                "num_procedures_x_time_in_hospital",
                "num_lab_procedures_x_time_in_hospital",
                "number_diagnoses_x_time_in_hospital",
                "age_x_number_diagnoses",
                "age_x_num_medications",
                "total_medications_x_number_diagnoses",
                "num_medications_x_num_procedures",
                "time_in_hospital_x_num_lab_procedures",
                "num_medications_x_num_lab_procedures",
                "change_x_num_medications",
                "num_medications_x_numchange",
            ],
            "ratio_features": [
                "procedure_medication_ratio",
                "lab_procedure_ratio",
                "diagnosis_procedure_ratio",
            ],
            "categorical_features": {
                "admission_type_id": list(range(1, 9)),
                "discharge_disposition_id": list(range(1, 27)),
                "admission_source_id": list(range(1, 26)),
                "level1_diag1": list(range(0, 9)),
            },
            "lab_features": {
                "a1cresult": {"mapping": {">7": 1, ">8": 1, "Norm": 0, "None": -99}},
                "max_glu_serum": {
                    "mapping": {">200": 1, ">300": 1, "Norm": 0, "None": -99}
                },
            },
        },
        "transformations": {
            "numeric_scaling": "standard",
            "outlier_handling": {"method": "clip", "std_multiplier": 5},
            "missing_values": {"numeric": "mean", "categorical": "mode"},
        },
        "target": {
            "name": "readmitted",
            "type": "binary",
            "mapping": {">30": 0, "<30": 1, "NO": 0},
        },
    }

    config_path = output_dir / "preprocessing_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Created preprocessing config at {config_path}")


def create_model_config(output_dir: Path, model_type: str) -> None:
    """Create config.json for the Hugging Face Inference API."""
    config = {
        "architectures": ["TabularBinaryClassification"],
        "model_type": model_type,
        "num_classes": 2,
        "id2label": {"0": "NO_READMISSION", "1": "READMISSION"},
        "label2id": {"NO_READMISSION": 0, "READMISSION": 1},
        "task_specific_params": {
            "classification": {"problem_type": "binary_classification"}
        },
        "preprocessing": {"featurization_config": "preprocessing_config.json"},
    }

    config_path = output_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Created model config at {config_path}")


def create_tokenizer_config(output_dir: Path) -> None:
    """Create tokenizer configuration for tabular data processing."""
    config = {
        "feature_extractor_type": "tabular",
        "framework": "pt",
        "num_features": None,  # Will be set during preprocessing
        "requires_preprocessing": True,
        "preprocessing_config": "preprocessing_config.json",
    }

    config_path = output_dir / "tokenizer_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Created tokenizer config at {config_path}")


def copy_model_artifacts(output_dir: Path) -> bool:
    """Copy all model artifacts to the output directory."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_type, paths in MODEL_DIRS.items():
            if paths["root"].exists():
                dest_dir = output_dir / model_type
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Copy model files
                model_dest = dest_dir / "model"
                model_dest.mkdir(parents=True, exist_ok=True)

                # Special handling for AutoGluon
                if model_type == "autogluon":
                    model_source = paths["model"].parent
                    if model_source.exists():
                        # Recursive copy for AutoGluon model directory
                        for root, dirs, files in os.walk(model_source):
                            relative_path = Path(root).relative_to(model_source)
                            (model_dest / relative_path).mkdir(
                                parents=True, exist_ok=True
                            )
                            for file in files:
                                src_file = Path(root) / file
                                dst_file = model_dest / relative_path / file
                                shutil.copy2(src_file, dst_file)
                else:
                    # Regular model file copying
                    if paths["model"].exists():
                        shutil.copy2(paths["model"], model_dest)
                        model_dir = paths["model"].parent
                        for file in model_dir.glob("*"):
                            if file.is_file():
                                shutil.copy2(file, model_dest)

                # Copy metrics
                metrics_dest = dest_dir / "metrics"
                metrics_dest.mkdir(parents=True, exist_ok=True)
                if paths["metrics"].exists():
                    shutil.copy2(paths["metrics"], metrics_dest)

                # Copy feature importance if it exists
                if (
                    "feature_importance" in paths
                    and paths["feature_importance"].exists()
                ):
                    shutil.copy2(
                        paths["feature_importance"],
                        metrics_dest / "feature_importance.csv",
                    )

                # Copy plots
                if paths["plots"].exists():
                    plots_dest = dest_dir / "plots"
                    plots_dest.mkdir(parents=True, exist_ok=True)
                    for plot in paths["plots"].glob("*.png"):
                        shutil.copy2(plot, plots_dest)

                logger.info(f"Copied {model_type} artifacts to {dest_dir}")

        return True
    except Exception as e:
        logger.error(f"Error copying model artifacts: {e}")
        return False


def find_best_model() -> Tuple[Optional[str], Optional[Dict]]:
    """Find the model with the highest test AUC score."""
    best_model = None
    best_metrics = None

    for model_type, paths in MODEL_DIRS.items():
        metrics = load_metrics(paths["metrics"])
        if metrics and (
            not best_metrics or metrics["test_auc"] > best_metrics["test_auc"]
        ):
            best_metrics = metrics
            best_model = model_type

    if best_model:
        logger.info(
            f"Best model is {best_model} with AUC: {best_metrics['test_auc']:.4f}"
        )
    else:
        logger.error("No valid model metrics found")

    return best_model, best_metrics


def create_model_card(best_model: str, metrics: Dict, output_dir: Path) -> None:
    """Create and save the model card."""
    from huggingface_hub.repocard_data import EvalResult

    eval_results = [
        EvalResult(
            task_type="binary-classification",
            dataset_type="hospital-readmission",
            dataset_name="Diabetes 130-US Hospitals",
            metric_type=metric_type,
            metric_value=metrics.get(f"test_{metric_type}", "N/A"),
            metric_name=metric_type,
        )
        for metric_type in ["accuracy", "auc"]
    ]

    card_data = ModelCardData(
        language="en", license="mit", model_name=REPO_ID, eval_results=eval_results
    )

    # Load feature importance if available
    feature_importance_df = None
    if "feature_importance" in MODEL_DIRS[best_model]:
        try:
            fi_path = MODEL_DIRS[best_model]["feature_importance"]
            if fi_path.exists():
                feature_importance_df = pd.read_csv(fi_path)
                feature_importance_df = feature_importance_df[
                    feature_importance_df["importance"] > 0
                ].sort_values("importance", ascending=False)
        except Exception as e:
            logger.warning(f"Could not load feature importance: {e}")

    content = f"""
---
{card_data.to_yaml()}
---

# {REPO_ID}

## Model Description

This model predicts 30-day hospital readmissions for diabetic patients using historical patient data
and machine learning techniques. The model aims to identify high-risk individuals enabling targeted
interventions and improved healthcare resource allocation.

## Overview

- **Task:** Binary Classification (Hospital Readmission Prediction)
- **Model Type:** {best_model}
- **Framework:** Python {best_model.replace('_', ' ').title()}
- **License:** MIT
- **Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Performance Metrics

- **Test Accuracy:** {metrics.get('test_accuracy', 'N/A'):.4f}
- **Test ROC-AUC:** {metrics.get('test_auc', 'N/A'):.4f}
"""

    # Add feature importance section if available
    if feature_importance_df is not None and not feature_importance_df.empty:
        content += "\n## Feature Importance\n\n"
        content += "Significant features and their importance scores:\n\n"
        content += "| Feature | Importance | p-value | 99% CI |\n"
        content += "|---------|------------|----------|----------|\n"

        for idx, row in feature_importance_df.iterrows():
            p_value = (
                f"{row['p_value']:.2e}"
                if row["p_value"] < 0.001
                else f"{row['p_value']:.4f}"
            )
            content += f"| {idx} | {row['importance']:.4f} | {p_value} | [{row['p99_low']:.4f}, {row['p99_high']:.4f}] |\n"

        content += "\n*Note: Only features with non-zero importance are shown. The confidence intervals (CI) are calculated at the 99% level. Features with p-value < 0.05 are considered statistically significant.*\n"

    content += """
## Features

### Numeric Features
- Patient demographics (age)
- Hospital stay metrics (time_in_hospital, num_procedures, num_lab_procedures)
- Medication metrics (num_medications, total_medications)
- Service utilization (number_outpatient, number_emergency, number_inpatient)
- Diagnostic information (number_diagnoses)

### Binary Features
- Patient characteristics (gender)
- Medication flags (diabetesmed, change, insulin_with_oral)

### Interaction Features
- Time-based interactions (medications × time, procedures × time)
- Complexity indicators (age × diagnoses, medications × procedures)
- Resource utilization (lab procedures × time, medications × changes)

### Ratio Features
- Resource efficiency (procedure/medication ratio, lab/procedure ratio)
- Diagnostic density (diagnosis/procedure ratio)

## Intended Use

This model is designed for healthcare professionals to assess the risk of 30-day readmission
for diabetic patients. It should be used as a supportive tool in conjunction with clinical judgment.

### Primary Intended Uses
- Predict likelihood of 30-day hospital readmission
- Support resource allocation and intervention planning
- Aid in identifying high-risk patients
- Assist in care management decision-making

### Out-of-Scope Uses
- Non-diabetic patient populations
- Predicting readmissions beyond 30 days
- Making final decisions without clinical oversight
- Use as sole determinant for patient care decisions
- Emergency or critical care decision-making

## Training Data

The model was trained on the [Diabetes 130-US Hospitals Dataset](https://doi.org/10.24432/C5230J)
(1999-2008) from UCI ML Repository. This dataset includes:

- Over 100,000 hospital admissions
- 50+ features including patient demographics, diagnoses, procedures
- Binary outcome: readmission within 30 days
- Comprehensive medication tracking
- Detailed hospital utilization metrics

## Training Procedure

### Data Preprocessing
- Missing value imputation using mean/mode
- Outlier handling using 5-sigma clipping
- Feature scaling using StandardScaler
- Categorical encoding using one-hot encoding
- Log transformation for skewed features

### Feature Engineering
- Created interaction terms between key variables
- Generated resource utilization ratios
- Aggregated medication usage metrics
- Developed time-based interaction features
- Constructed diagnostic density metrics

### Model Training
- Data split: 70% training, 15% validation, 15% test
- Cross-validation for model selection
- Hyperparameter optimization via grid search
- Early stopping to prevent overfitting
- Model selection based on ROC-AUC performance

## Limitations & Biases

### Known Limitations
- Model performance depends on data quality and completeness
- Limited to the scope of training data timeframe (1999-2008)
- May not generalize to significantly different healthcare systems
- Requires standardized input data format

### Potential Biases
- May exhibit demographic biases present in training data
- Performance may vary across different hospital systems
- Could be influenced by regional healthcare practices
- Might show temporal biases due to historical data

### Recommendations
- Regular model monitoring and retraining
- Careful validation in new deployment contexts
- Assessment of performance across demographic groups
- Integration with existing clinical workflows

## Monitoring & Maintenance

### Monitoring Requirements
- Track prediction accuracy across different patient groups
- Monitor input data distribution shifts
- Assess feature importance stability
- Evaluate performance metrics over time

### Maintenance Schedule
- Quarterly performance reviews recommended
- Annual retraining with updated data
- Regular bias assessments
- Ongoing validation against current practices

## Citation

```bibtex
@misc{diabetes-readmission-model,
  title = {Hospital Readmission Prediction Model for Diabetic Patients},
  author = {Agustin, Jonathan and Robertson, Zack and Vo, Lisa},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/{REPO_ID}}}
}

@misc{diabetes-dataset,
  title = {Diabetes 130-US Hospitals for Years 1999-2008 Data Set},
  author = {Strack, B. and DeShazo, J. and Gennings, C. and Olmo, J. and
            Ventura, S. and Cios, K. and Clore, J.},
  year = {2014},
  publisher = {UCI Machine Learning Repository},
  doi = {10.24432/C5230J}
}
```

## Model Card Authors

Jonathan Agustin, Zack Robertson, Lisa Vo

## For Questions, Issues, or Feedback

- GitHub Issues: [Repository Issues](https://github.com/aai540-group3/diabetes-readmission/issues)
- Email: [team contact information]

## Updates and Versions

- {pd.Timestamp.now().strftime('%Y-%m-%d')}: Initial model release
- Feature engineering pipeline implemented
- Comprehensive preprocessing system added
- Model evaluation and selection completed

---
Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}
"""

    card = ModelCard(content)
    card.save(output_dir / "README.md")
    logger.info(f"Created model card at {output_dir / 'README.md'}")


def upload_to_huggingface() -> bool:
    """Upload the model directory to Hugging Face."""
    token = get_hf_token()
    os.environ["HF_TRANSFER"] = "1"  # Enable faster uploads

    try:
        # Login to Hugging Face
        logger.info("Logging into Hugging Face...")
        login_cmd = ["huggingface-cli", "login", "--token", token]
        subprocess.run(login_cmd, check=True, capture_output=True, text=True)

        # Try to create repo
        logger.info(f"Creating repo {REPO_ID} if it doesn't exist...")
        repo_create_cmd = [
            "huggingface-cli",
            "repo",
            "create",
            "diabetes-readmission",
            "--type",
            "model",
            "--organization",
            "aai540-group3",
            "--yes",
        ]
        result = subprocess.run(repo_create_cmd, capture_output=True, text=True)
        if (
            result.returncode != 0
            and "already created this model repo" not in result.stderr
        ):
            logger.error(f"Failed to create repo: {result.stderr}")
            return False
        logger.info("Repository ready for upload")

        # Upload command
        logger.info(f"Starting upload to {REPO_ID}...")
        upload_cmd = [
            "huggingface-cli",
            "upload",
            REPO_ID,
            str(OUTPUT_DIR),
            "--repo-type",
            "model",
        ]
        subprocess.run(upload_cmd, check=True, text=True)
        logger.info(f"Successfully uploaded model to {REPO_ID}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error uploading to Hugging Face: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        return False


def main():
    """Main deployment function."""
    try:
        # Copy all model artifacts to output directory
        if not copy_model_artifacts(OUTPUT_DIR):
            logger.error("Failed to copy model artifacts")
            return False

        # Find best performing model
        best_model, best_metrics = find_best_model()
        if not best_model or not best_metrics:
            logger.error("Could not determine best model")
            return False

        # Create model card
        create_model_card(best_model, best_metrics, OUTPUT_DIR)

        # Create inference configs
        create_preprocessing_config(OUTPUT_DIR, best_model)
        create_model_config(OUTPUT_DIR, best_model)
        create_tokenizer_config(OUTPUT_DIR)

        # Upload to Hugging Face
        if not upload_to_huggingface():
            logger.error("Failed to upload to Hugging Face")
            return False

        logger.info("Deployment completed successfully")
        return True

    except Exception as e:
        logger.exception(f"Deployment failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
