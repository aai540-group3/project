---
language: en
license: mit
model-index:
- name: aai540-group3/diabetes-readmission
  results:
  - task:
      type: binary-classification
    dataset:
      name: Diabetes 130-US Hospitals
      type: hospital-readmission
    metrics:
    - type: accuracy
      value: 0.8865474882652552
      name: accuracy
    - type: auc
      value: 0.6467403398083669
      name: auc
---

# aai540-group3/diabetes-readmission

## Model Description

This model predicts 30-day hospital readmissions for diabetic patients using historical patient data
and machine learning techniques. The model aims to identify high-risk individuals enabling targeted
interventions and improved healthcare resource allocation.

## Overview

- **Task:** Binary Classification (Hospital Readmission Prediction)
- **Model Type:** autogluon
- **Framework:** Python Autogluon
- **License:** MIT
- **Last Updated:** 2024-10-29

## Performance Metrics

- **Test Accuracy:** 0.8865
- **Test ROC-AUC:** 0.6467

## Feature Importance

Significant features and their importance scores:

| Feature | Importance | p-value | 99% CI |
|---------|------------|----------|----------|
| 0 | 0.0563 | 3.24e-04 | [0.0294, 0.0832] |
| 1 | 0.0358 | 8.45e-06 | [0.0290, 0.0426] |
| 2 | 0.0080 | 0.0083 | [-0.0013, 0.0173] |
| 3 | 0.0046 | 1.96e-04 | [0.0027, 0.0065] |
| 4 | 0.0023 | 0.0055 | [-0.0001, 0.0046] |
| 5 | 0.0008 | 0.1840 | [-0.0027, 0.0043] |

*Note: Only features with non-zero importance are shown. The confidence intervals (CI) are calculated at the 99% level. Features with p-value < 0.05 are considered statistically significant.*


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
  howpublished = {\url{https://huggingface.co/{REPO_ID}}}
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
