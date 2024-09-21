[![Deploy HF Space: OpenAI TTS](https://github.com/aai540-group3/project/actions/workflows/deploy-tts-space.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/deploy-tts-space.yml)

# Final Project

## General Project Repository Organization

```
project/
├── .github
│   └── workflows                    # GitHub Actions workflows
├── LICENSE                          # Project license
├── README.md                        # Project README
├── Makefile                         # Makefile with commands like `make data` or `make train`
│
├── configs                          # Configuration files (models and training hyperparameters)
│   └── model1.yaml
│
├── data                             # Data directory
│   ├── external                     # Data from third-party sources
│   ├── interim                      # Intermediate data that has been transformed
│   ├── processed                    # The final, canonical datasets for modeling
│   └── raw                          # The original, immutable data dump
│
├── docs                             # Project documentation
│   └── requirements.md              # Project requirements and specifications
│
├── huggingface                      # Hugging Face applications
│
├── models                           # Trained and serialized models
│
├── notebooks                        # Jupyter notebooks
│
├── references                       # Data dictionaries, manuals, and other explanatory materials
│
├── reports                          # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                      # Generated graphics and figures for reporting
│
├── requirements.txt                 # Requirements file for reproducing the analysis environment
│
├── scripts                          # Scripts for various tasks (e.g., setup, deployment)
│
├── src                              # Source code for use in this project
│   ├── __init__.py                  # Makes `src` a Python module
│   │
│   ├── data                         # Data engineering scripts
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   ├── cleaning.py
│   │   ├── ingestion.py
│   │   ├── labeling.py
│   │   ├── splitting.py
│   │   └── validation.py
│   │
│   ├── models                       # ML model engineering (a folder for each model)
│   │   └── model1
│   │       ├── __init__.py
│   │       ├── dataloader.py
│   │       ├── hyperparameters_tuning.py
│   │       ├── model.py
│   │       ├── predict.py
│   │       ├── preprocessing.py
│   │       └── train.py
│   │
│   └── visualization                # Scripts to create visualizations
│       ├── __init__.py
│       ├── evaluation.py
│       └── exploration.py
│
└── terraform                        # Infrastructure as code using Terraform
```
