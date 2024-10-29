# Diabetic Readmission Risk Prediction

---

## Status

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![release](https://img.shields.io/github/release/aai540-group3/project.svg)](https://github.com/aai540-group3/project/releases)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/66d984def1ee4ae481e78b91ffd159f0)](https://app.codacy.com/gh/aai540-group3/project/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![CodeQL Advanced](https://github.com/aai540-group3/project/actions/workflows/codeql.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/codeql.yml) [![Codacy Security Scan](https://github.com/aai540-group3/project/actions/workflows/codacy-analysis.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/codacy-analysis.yml) [![Dependabot Updates](https://github.com/aai540-group3/project/actions/workflows/dependabot/dependabot-updates.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/dependabot/dependabot-updates.yml)

[![MLOps Pipeline](https://github.com/aai540-group3/project/actions/workflows/mlops-pipeline.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/mlops-pipeline.yml) [![Generate Video](https://github.com/aai540-group3/project/actions/workflows/generate-video.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/generate-video.yml) [![Generate PDFs](https://github.com/aai540-group3/project/actions/workflows/generate-pdfs.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/generate-pdfs.yml) [![Package Deliverables](https://github.com/aai540-group3/project/actions/workflows/package-deliverables.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/package-deliverables.yml) [![Create Release](https://github.com/aai540-group3/project/actions/workflows/create-release.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/create-release.yml) [![Deploy HF Space: OpenAI TTS](https://github.com/aai540-group3/project/actions/workflows/deploy-tts-space.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/deploy-tts-space.yml)

---

## Table of Contents

- [Diabetic Readmission Risk Prediction](#diabetic-readmission-risk-prediction)
  - [Status](#status)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Problem Statement](#problem-statement)
  - [Impact Measurement](#impact-measurement)
    - [Model Performance Metrics](#model-performance-metrics)
    - [Reduction in Readmission Rates](#reduction-in-readmission-rates)
    - [Cost Savings Analysis](#cost-savings-analysis)
    - [Resource Optimization](#resource-optimization)
  - [Solution Overview](#solution-overview)
    - [System Architecture](#system-architecture)
    - [Data Sources](#data-sources)
    - [Data Engineering](#data-engineering)
    - [Feature Engineering](#feature-engineering)
    - [Model Training and Evaluation](#model-training-and-evaluation)
    - [Model Deployment](#model-deployment)
    - [Model Monitoring](#model-monitoring)
    - [Continuous Integration and Continuous Deployment (CI/CD)](#continuous-integration-and-continuous-deployment-cicd)
  - [Repository Structure](#repository-structure)
  - [Conclusion](#conclusion)
  - [References](#references)
  - [License](#license)

---

## Overview

Hospital readmissions are a critical issue affecting healthcare systems globally. Frequent readmissions, particularly within 30 days of discharge, not only escalate healthcare costs but also negatively impact patient satisfaction and clinical outcomes. Among chronic diseases, diabetes mellitus is prevalent and poses significant management challenges, leading to higher readmission rates. Effective strategies to predict and reduce readmissions among diabetic patients are essential for enhancing patient care and optimizing healthcare resources.

This project aims to develop a robust machine learning (ML) system that predicts 30-day hospital readmissions among diabetic patients. By leveraging historical patient data and advanced analytical techniques, the system seeks to identify high-risk individuals, enabling proactive interventions and contributing to improved healthcare efficiency and patient well-being.

---

## Problem Statement

The rising rate of hospital readmissions among diabetic patients presents a multifaceted problem for healthcare providers. It strains resources, increases healthcare expenditures, and can negatively impact patient health and quality of life. The complexity of diabetes management, potential complications, and the need for strict adherence to treatment regimens contribute to the elevated readmission risk for this population.

This project addresses the challenge of accurately predicting 30-day hospital readmissions among diabetic patients. By developing a robust and scalable machine learning system, we aim to analyze a wide range of patient attributes—including demographics, medical history, treatment patterns, and hospital procedures—to identify individuals at high risk of readmission. This predictive capability will empower clinicians to implement targeted interventions, potentially reducing readmission rates and improving overall patient care.

---

## Impact Measurement

To assess the effectiveness of the ML system, we will employ a comprehensive set of metrics and analyses. These measures are designed to evaluate not only the technical performance of the model but also its real-world impact on healthcare delivery.

### Model Performance Metrics

The predictive accuracy and reliability of the model are critical for its adoption in clinical settings. We will utilize the following standard classification metrics:

- **Accuracy**: Measures the proportion of correct predictions out of all predictions.
- **Precision**: Assesses the model's ability to correctly identify true positive cases among all positive predictions.
- **Recall (Sensitivity)**: Evaluates the model's capacity to identify all actual positive cases.
- **F1-Score**: Represents the harmonic mean of precision and recall, providing a balance between the two.
- **ROC-AUC Score**: Quantifies the model's ability to distinguish between classes by plotting the true positive rate against the false positive rate.

These metrics will be calculated on a held-out test dataset to provide an unbiased evaluation of the model's performance.

### Reduction in Readmission Rates

A key goal of deploying the ML system is to achieve a measurable reduction in readmission rates among diabetic patients. We plan to:

- **Establish a Baseline**: Determine the current readmission rate prior to implementing the ML system.
- **Post-Implementation Monitoring**: Track readmission rates following deployment to assess changes.
- **Target Reduction**: Aim for at least a 10% decrease in 30-day readmission rates, indicating a significant positive impact.

Comparing these rates before and after implementation will help us assess the practical benefits of the system in a real-world healthcare environment.

### Cost Savings Analysis

Reducing readmissions can result in substantial cost savings for healthcare institutions. We will analyze:

- **Direct Cost Savings**: Calculate the reduction in expenses associated with fewer hospital stays, treatments, and procedures for readmitted patients.
- **Indirect Cost Savings**: Consider savings from improved resource allocation, such as reduced burden on hospital staff and facilities.
- **Return on Investment (ROI)**: Evaluate the financial benefits in relation to the costs incurred in developing and deploying the ML system.

This analysis will provide insights into the economic value of the system for healthcare providers.

### Resource Optimization

Efficient utilization of healthcare resources is essential for enhancing patient care while controlling costs. We will examine:

- **Bed Occupancy Rates**: Assess changes in bed availability and whether reduced readmissions lead to better management of hospital capacity.
- **Staff Workload**: Evaluate the impact on healthcare provider workloads, aiming for a more balanced distribution of patient care duties.
- **Preventive Care Allocation**: Observe whether resources can be reallocated towards preventive measures and patient education due to reduced readmissions.

By monitoring these indicators, we can validate the system's contribution to overall healthcare efficiency and patient care quality.

---

## Solution Overview

Our proposed solution integrates robust data handling, advanced modeling techniques, seamless deployment strategies, and continuous monitoring processes. Each component is designed to contribute to an effective and reliable ML system for predicting hospital readmissions.

### System Architecture

The system comprises multiple components, each responsible for specific tasks, integrated to function cohesively:

1. **Data Ingestion**: Fetching and storing raw data from reliable sources.
2. **Data Engineering**: Cleaning, preprocessing, and transforming data to prepare for modeling.
3. **Feature Engineering**: Creating new features to capture underlying patterns and relationships.
4. **Model Training and Evaluation**: Developing predictive models and assessing their performance.
5. **Model Deployment**: Containerizing and deploying models as RESTful APIs.
6. **Model Monitoring**: Tracking model performance over time and detecting data drift.
7. **CI/CD Pipeline**: Automating the development workflow using DVC and GitHub Actions.

### Data Sources

We use the **Diabetes 130-US hospitals for years 1999-2008 Data Set** from the UCI Machine Learning Repository.

- **Size**: 101,766 patient records.
- **Features**: 55 attributes including demographics, medical history, lab results, medications, and hospitalization details.
- **Target Variable**: `readmitted` indicator (0 or 1) for 30-day readmission.

### Data Engineering

- **Ingestion**: Using `ingest.py` to fetch and store the dataset in CSV and Parquet formats.
- **Cleaning**: Applying `preprocess.py` to handle duplicates, missing values, inconsistent data types, and irrelevant features.
- **Feature Store Integration**: Utilizing Feast to manage feature definitions and data consistency.

### Feature Engineering

- **New Features**: Engineered features like `total_medications`, `total_encounters`, `procedures_per_day`, and others to enhance model performance.
- **Log Transformations**: Applied to skewed features to reduce skewness.
- **One-Hot Encoding**: Converted categorical variables into numerical format.

### Model Training and Evaluation

- **Logistic Regression**: A baseline model trained with standardized features and addressing class imbalance using SMOTE.
- **AutoGluon TabularPredictor**: Automated machine learning tool that explores various models and hyperparameters.
- **Evaluation Metrics**: Models evaluated on accuracy, precision, recall, F1-score, and ROC-AUC.

### Model Deployment

- **Serialization and Versioning**: Models saved and versioned using DVC.
- **Deployment Strategy**: Containerizing models with Docker and orchestrating deployment using Kubernetes or similar tools.
- **Endpoint Exposure**: Models deployed as RESTful APIs using frameworks like FastAPI.

### Model Monitoring

- **DVCLive Integration**: Real-time tracking of metrics and parameters during training.
- **DVC Studio**: Interactive dashboards for experiment management.
- **Performance Monitoring**: Alerts and notifications set up to detect performance degradation.

### Continuous Integration and Continuous Deployment (CI/CD)

- **DVC Pipeline Integration**: Entire ML workflow orchestrated using DVC and defined in `dvc.yaml`.
- **GitHub Actions Workflow**: Automates the development process, integrating with DVC for data and model versioning.
- **Pipeline Stages**:

  1. **Setup**: Environment preparation.
  2. **Ingestion**: Data fetching.
  3. **Preprocessing**: Data cleaning and preparation.
  4. **Exploration**: Exploratory data analysis.
  5. **Feature Engineering**: Creating and registering features.
  6. **Model Training**: Training multiple models.

---

## Repository Structure

```plaintext
project/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml      # CI/CD workflow configuration
├── configs/                        # Configuration files for the project
├── data/                           # Data storage
│   ├── raw/                        # Original, unprocessed data
│   ├── interim/                    # Intermediate data after preprocessing
│   └── processed/                  # Data ready for model training
├── models/                         # Trained model artifacts
│   ├── logistic_regression/
│   ├── neural_network/
│   └── autogluon/
├── reports/                        # Reports and analysis outputs
│   ├── plots/
│   └── metrics/
├── src/                            # Source code for the project
│   ├── ingest.py                   # Data ingestion script
│   ├── preprocess.py               # Data preprocessing script
│   ├── explore.py                  # Exploratory data analysis
│   ├── featurize.py                # Feature engineering script
│   ├── utils.py                    # Utility functions
│   └── models/                     # Model training scripts
│       ├── logistic_regression/
│       ├── neural_network/
│       └── autogluon/
├── dvc.yaml                        # DVC pipeline configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── LICENSE                         # License file
```

---

## Conclusion

This project presents a comprehensive machine learning system aimed at predicting 30-day hospital readmissions for diabetic patients. By integrating advanced data engineering practices, sophisticated modeling techniques, and robust deployment strategies, the proposed solution is poised to make a significant impact in healthcare settings.

Emphasis on security, ethical considerations, and bias mitigation ensures that the system not only performs effectively but also responsibly. The use of tools like DVC and Feast enhances reproducibility and scalability. Additionally, automated CI/CD pipelines facilitate continuous improvement and collaboration.

Moving forward, collaboration with healthcare professionals will be crucial to refine the system, ensuring that it integrates seamlessly into clinical workflows and truly enhances patient care.

---

## References

1. **UCI Machine Learning Repository**: ["Diabetes 130-US hospitals for years 1999-2008 Data Set"](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
2. **Health Insurance Portability and Accountability Act (HIPAA)**: [U.S. Department of Health & Human Services](https://www.hhs.gov/hipaa/index.html)
3. **American Medical Association (AMA) Code of Medical Ethics**: [AMA Ethical Guidelines](https://www.ama-assn.org/delivering-care/ethics/code-medical-ethics-overview)
4. **AutoGluon Documentation**: [AutoGluon TabularPredictor](https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html)
5. **DVC (Data Version Control)**: [DVC Documentation](https://dvc.org/doc)
6. **Feast**: [Feast Feature Store](https://feast.dev/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.