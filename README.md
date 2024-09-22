# Final Project

[![Deploy HF Space: OpenAI TTS](https://github.com/aai540-group3/project/actions/workflows/deploy-tts-space.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/deploy-tts-space.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## General Project Repository Organization

```text
project/
├── .github
│   └── workflows                    # GitHub Actions workflows
├── LICENSE                          # Project license
├── README.md                        # Project README
├── Makefile                         # Makefile with commands like `make data` or `make train`
│
├── conf                          # Configuration files (models and training hyperparameters)
│   └── model1.yaml
│
├── data                             # Data directory
│   ├── external                     # Data from third-party sources
│   ├── interim                      # Intermediate data that has been transformed
│   ├── processed                    # The final, canonical datasets for modeling
│   └── raw                          # The original, immutable data dump
│
├── docs                             # Project documentation
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

## WSL Ray fix

<https://github.com/ray-project/ray/issues/45492>

```bash
sed -i '
/def get_current_node_accelerator_type/,/return cuda_device_type/ c\
    @staticmethod\
    def get_current_node_accelerator_type() -> Optional[str]:\
        import ray._private.thirdparty.pynvml as pynvml\
\
        try:\
            pynvml.nvmlInit()\
        except pynvml.NVMLError:\
            return None  # pynvml init failed\
        device_count = pynvml.nvmlDeviceGetCount()\
        cuda_device_type = None\
        if device_count > 0:\
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)\
            device_name = pynvml.nvmlDeviceGetName(handle)\
            if isinstance(device_name, bytes):\
                try:\
                    device_name = device_name.decode("utf-16be")\
                except UnicodeDecodeError as e:\
                    device_name = device_name.decode("utf-8")\
            cuda_device_type = (\
                NvidiaGPUAcceleratorManager._gpu_name_to_accelerator_type(device_name)\
            )\
        pynvml.nvmlShutdown()\
        return cuda_device_type
' .venv/lib/python3.11/site-packages/ray/_private/accelerators/nvidia_gpu.py
```
