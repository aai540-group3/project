# Software Requirements Specification (SRS)

## Document Version

1.0

## Date

September 24, 2024

## Project Name

AAI-540 Final Project - Machine Learning System for Predicting Hospital Readmissions in Diabetic Patients

---

## Table of Contents

- [Software Requirements Specification (SRS)](#software-requirements-specification-srs)
  - [Document Version](#document-version)
  - [Date](#date)
  - [Project Name](#project-name)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Purpose](#11-purpose)
    - [1.2 Scope](#12-scope)
    - [1.3 Definitions, Acronyms, and Abbreviations](#13-definitions-acronyms-and-abbreviations)
    - [1.4 References](#14-references)
    - [1.5 Overview](#15-overview)
  - [2. Overall Description](#2-overall-description)
    - [2.1 Product Perspective](#21-product-perspective)
    - [2.2 Product Functions](#22-product-functions)
    - [2.3 User Classes and Characteristics](#23-user-classes-and-characteristics)
    - [2.4 Operating Environment](#24-operating-environment)
    - [2.5 Design and Implementation Constraints](#25-design-and-implementation-constraints)
    - [2.6 Assumptions and Dependencies](#26-assumptions-and-dependencies)
  - [3. Specific Requirements](#3-specific-requirements)
    - [3.1 Functional Requirements](#31-functional-requirements)
      - [3.1.1 Data Ingestion and Storage](#311-data-ingestion-and-storage)
      - [3.1.2 Data Processing](#312-data-processing)
      - [3.1.3 Feature Engineering](#313-feature-engineering)
      - [3.1.4 Model Development](#314-model-development)
      - [3.1.5 Model Evaluation](#315-model-evaluation)
      - [3.1.6 Model Deployment](#316-model-deployment)
      - [3.1.7 Model Monitoring](#317-model-monitoring)
      - [3.1.8 CI/CD Pipeline](#318-cicd-pipeline)
      - [3.1.9 Infrastructure Management](#319-infrastructure-management)
      - [3.1.10 Security and Access Control](#3110-security-and-access-control)
      - [3.1.11 Code Quality and Documentation](#3111-code-quality-and-documentation)
    - [3.2 Non-functional Requirements](#32-non-functional-requirements)
      - [3.2.1 Performance Requirements](#321-performance-requirements)
      - [3.2.2 Security Requirements](#322-security-requirements)
      - [3.2.3 Usability Requirements](#323-usability-requirements)
      - [3.2.4 Maintainability Requirements](#324-maintainability-requirements)
    - [3.3 External Interface Requirements](#33-external-interface-requirements)
      - [3.3.1 User Interfaces](#331-user-interfaces)
      - [3.3.2 Hardware Interfaces](#332-hardware-interfaces)
      - [3.3.3 Software Interfaces](#333-software-interfaces)
      - [3.3.4 Communications Interfaces](#334-communications-interfaces)
  - [4. Project Deliverables](#4-project-deliverables)
    - [4.1 Overview](#41-overview)
    - [4.2 Deliverables List](#42-deliverables-list)
    - [4.3 Deliverable Descriptions](#43-deliverable-descriptions)
      - [D-01: ML System Design Document](#d-01-ml-system-design-document)
      - [D-02: Video Demonstration](#d-02-video-demonstration)
      - [D-03: Video Demonstration Outline/Transcript](#d-03-video-demonstration-outlinetranscript)
      - [D-04: Codebase GitHub Repository](#d-04-codebase-github-repository)
  - [5. Appendices](#5-appendices)
    - [5.1 Appendix A: Glossary of Terms](#51-appendix-a-glossary-of-terms)
    - [5.2 Appendix B: References](#52-appendix-b-references)

---

## 1. Introduction

### 1.1 Purpose

The purpose of this Software Requirements Specification (SRS) document is to provide a detailed description of the requirements for the development of a machine learning system that predicts 30-day hospital readmissions among diabetic patients. This document outlines the system's functional and non-functional requirements, design constraints, and provides necessary information for the development and testing teams to understand the project's scope and objectives.

### 1.2 Scope

The system to be developed is a comprehensive machine learning pipeline that includes data ingestion, data engineering, feature engineering, model training, evaluation, deployment, and monitoring. The system aims to:

- Predict the likelihood of 30-day hospital readmissions for diabetic patients.
- Provide actionable insights to healthcare providers to reduce readmission rates.
- Be scalable, maintainable, and adhere to MLOps best practices.

The project includes the development of:

- Data processing scripts.
- Machine learning models using Scikit-learn and AutoGluon.
- Deployment of models via Hugging Face Spaces.
- CI/CD pipelines using GitHub Actions.
- Model monitoring using DVCLive and DVC Studio.
- Infrastructure management via Terraform.

### 1.3 Definitions, Acronyms, and Abbreviations

- **ML**: Machine Learning
- **MLOps**: Machine Learning Operations
- **CI/CD**: Continuous Integration/Continuous Deployment
- **API**: Application Programming Interface
- **DVC**: Data Version Control
- **DVCLive**: DVC Live Logging Library
- **SRS**: Software Requirements Specification
- **AWS**: Amazon Web Services

### 1.4 References

- [IEEE Std 830-1998: IEEE Recommended Practice for Software Requirements Specifications](https://ieeexplore.ieee.org/document/720574)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [AutoGluon Documentation](https://auto.gluon.ai/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Hugging Face Datasets](https://huggingface.co/datasets/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [DVC Documentation](https://dvc.org/doc)
- [Terraform Documentation](https://www.terraform.io/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

### 1.5 Overview

This SRS document provides an in-depth look at the system to be developed, including its functionality, constraints, and requirements. It is structured to guide the development team through the project's needs and to ensure that all stakeholders have a clear understanding of the system's capabilities and limitations.

---

## 2. Overall Description

### 2.1 Product Perspective

The machine learning system is an independent project developed from scratch. It will leverage open-source technologies and align with MLOps best practices. The system addresses the problem of predicting 30-day hospital readmissions among diabetic patients, aiming to assist healthcare providers in making informed decisions to reduce readmission rates.

### 2.2 Product Functions

The system will perform the following functions:

- **Data Ingestion**: Load datasets from Hugging Face Datasets and store them securely.
- **Data Processing**: Clean and preprocess data using Pandas.
- **Feature Engineering**: Generate new features to enhance model performance.
- **Model Training**: Train models using Scikit-learn (Logistic Regression) and AutoGluon.
- **Model Evaluation**: Evaluate models using standard metrics and visualizations.
- **Model Deployment**: Deploy models as API endpoints using Hugging Face Spaces.
- **Model Monitoring**: Monitor model performance using DVCLive and DVC Studio.
- **CI/CD Pipeline**: Automate workflows using GitHub Actions.
- **Infrastructure Management**: Use Terraform for infrastructure provisioning.

### 2.3 User Classes and Characteristics

- **Data Scientists**: Users who will develop and train machine learning models.
- **ML Engineers**: Users responsible for deploying and maintaining the ML system's infrastructure.
- **Healthcare Providers**: End-users who will utilize the predictions to improve patient care.
- **Project Managers**: Users who will oversee project progress and ensure alignment with goals.

### 2.4 Operating Environment

- **Development Environment**: Local machines with Python 3.x and necessary libraries.
- **Deployment Environment**: Hosted on cloud platforms like Hugging Face Spaces and AWS S3.
- **Supported Operating Systems**: Linux, Windows, macOS.

### 2.5 Design and Implementation Constraints

- **Programming Language**: Python 3.x.
- **Libraries and Frameworks**: Pandas, Scikit-learn, AutoGluon, DVC, DVCLive.
- **Tools**: Git, GitHub, GitHub Actions, Terraform, Codacy.
- **Data Privacy**: Compliance with data protection regulations; use of anonymized data.
- **Resource Limitations**: Use of AWS Free Tier services to minimize costs.

### 2.6 Assumptions and Dependencies

- **Team Expertise**: Assumes team members are proficient in Python and familiar with ML tools.
- **Data Availability**: Assumes access to the required datasets without restrictions.
- **Third-Party Services**: Reliance on availability and stability of external services like GitHub, AWS, and Hugging Face.

---

## 3. Specific Requirements

### 3.1 Functional Requirements

#### 3.1.1 Data Ingestion and Storage

- **FR-01**: The system shall ingest data from Hugging Face Datasets.
- **FR-02**: The system shall store data securely in AWS S3.

#### 3.1.2 Data Processing

- **FR-03**: The system shall perform data cleaning to handle missing values and duplicates.
- **FR-04**: The system shall preprocess data for modeling using Pandas.

#### 3.1.3 Feature Engineering

- **FR-05**: The system shall create new features to improve model performance.
- **FR-06**: The system shall perform feature scaling and encoding using Scikit-learn.

#### 3.1.4 Model Development

- **FR-07**: The system shall train a Logistic Regression model using Scikit-learn.
- **FR-08**: The system shall utilize AutoGluon for automated model training and selection.

#### 3.1.5 Model Evaluation

- **FR-09**: The system shall evaluate models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
- **FR-10**: The system shall generate evaluation reports and visualizations.

#### 3.1.6 Model Deployment

- **FR-11**: The system shall deploy models as real-time API endpoints using Hugging Face Spaces.

#### 3.1.7 Model Monitoring

- **FR-12**: The system shall monitor model performance over time using DVCLive.
- **FR-13**: The system shall track experiments and model versions using DVC and DVC Studio.

#### 3.1.8 CI/CD Pipeline

- **FR-14**: The system shall implement CI/CD pipelines using GitHub Actions.
- **FR-15**: The system shall automate testing, building, and deployment processes.

#### 3.1.9 Infrastructure Management

- **FR-16**: The system shall use Terraform scripts to provision and manage infrastructure.

#### 3.1.10 Security and Access Control

- **FR-17**: The system shall manage secrets using GitHub Secrets.
- **FR-18**: The system shall restrict data and resource access to authorized users only.

#### 3.1.11 Code Quality and Documentation

- **FR-19**: The system shall enforce code quality standards using tools like Codacy, Flake8, and Pylint.
- **FR-20**: The system shall include comprehensive documentation, including an ML System Design Document.

### 3.2 Non-functional Requirements

#### 3.2.1 Performance Requirements

- **NFR-01**: The system shall provide predictions within acceptable timeframes for real-time use.
- **NFR-02**: The system shall efficiently handle the processing of large datasets.

#### 3.2.2 Security Requirements

- **NFR-03**: The system shall ensure data privacy and comply with relevant regulations.
- **NFR-04**: The system shall protect against unauthorized access and data breaches.

#### 3.2.3 Usability Requirements

- **NFR-05**: The system shall provide clear and comprehensive documentation for users and developers.
- **NFR-06**: The deployed APIs shall have well-defined interfaces and documentation.

#### 3.2.4 Maintainability Requirements

- **NFR-07**: The system's codebase shall be modular to facilitate easy updates and maintenance.
- **NFR-08**: The system shall include automated tests to detect regressions.

### 3.3 External Interface Requirements

#### 3.3.1 User Interfaces

- The system shall not require a graphical user interface but should provide command-line tools and scripts for interaction.

#### 3.3.2 Hardware Interfaces

- The system shall operate on standard hardware capable of running Python and the necessary libraries.

#### 3.3.3 Software Interfaces

- The system shall interface with external services such as AWS S3, GitHub, and Hugging Face Spaces through their APIs.

#### 3.3.4 Communications Interfaces

- The system shall use secure protocols (e.g., HTTPS) for all network communications.

---

## 4. Project Deliverables

### 4.1 Overview

The project includes several deliverables that demonstrate the system's capabilities and ensure compliance with course requirements. These deliverables include documentation, demonstration materials, and the complete codebase.

### 4.2 Deliverables List

| **ID** | **Deliverable**                               | **Description**                                                                               |
|--------|-----------------------------------------------|-----------------------------------------------------------------------------------------------|
| D-01   | ML System Design Document                     | A comprehensive document detailing the system's design, implementation, and analysis.         |
| D-02   | Video Demonstration                           | A 10-15 minute video showcasing the system's operation, architecture, and components.         |
| D-03   | Video Demonstration Outline/Transcript        | An outline, slide notes, or transcript accompanying the video demonstration.                 |
| D-04   | Codebase GitHub Repository                    | A professional and well-documented codebase stored on GitHub.                                 |

### 4.3 Deliverable Descriptions

#### D-01: ML System Design Document

**Description:**

- A finalized document that includes:
  - **Problem Statement:** Clearly defined machine learning problem.
  - **Impact Measurement:** Description of how the project's impact will be measured.
  - **Security Checklist and Risk Analysis:** Addressing sensitive data, bias, and ethical concerns.
  - **Solution Overview:** Detailed explanation of implementation, including:
    - Data Sources
    - Data Engineering
    - Feature Engineering
    - Model Training & Evaluation
    - Model Deployment
    - Model Monitoring
    - CI/CD Processes

**Requirements Met:**

- **FR-20**: Inclusion of comprehensive documentation.
- Supports communication with stakeholders and provides a technical reference.

#### D-02: Video Demonstration

**Description:**

- A 10-15 minute video that includes:
  - Introduction to the business use case.
  - Architecture diagrams explaining the system design.
  - Demonstration of system components in action.
  - Discussion of future improvements and challenges.
  - Demonstration of:
    - Feature store and feature groups.
    - Infrastructure monitoring dashboards.
    - Model and data monitoring reports.
    - CI/CD pipeline in successful and failed states.
    - Model registry.
    - Outputs of batch inference jobs or endpoint invocation.

#### D-03: Video Demonstration Outline/Transcript

**Description:**

- An accompanying document to the video that includes:
  - Outline of the video content.
  - Slide notes, if used.
  - Transcript of the narration.

**Requirements Met:**

- Enhances accessibility and provides a reference for the demonstration.
- Supports evaluation and review processes.

#### D-04: Codebase GitHub Repository

**Description:**

- A clean, organized, and professional repository that includes:
  - All source code, stored in appropriate formats (e.g., `.py`, `.ipynb`).
  - Documentation and comments within the code.
  - Data storage details and instructions.
  - Graphs and visualizations included within notebooks.
  - Clear commit history demonstrating team contributions.

**Requirements Met:**

- **FR-12**: Use of Git and GitHub for version control.
- **FR-19**: Code quality standards enforcement.
- **NFR-07**: Maintainability through modular code and documentation.

---

## 5. Appendices

### 5.1 Appendix A: Glossary of Terms

- **AutoGluon**: An open-source AutoML toolkit for automated machine learning.
- **AWS S3**: Amazon Web Services Simple Storage Service for scalable cloud storage.
- **Codacy**: A tool for automated code quality reviews and code analysis.
- **Flake8**: A tool for style guide enforcement and linting in Python.
- **Hugging Face**: A company providing open-source libraries for natural language processing.
- **Pylint**: A source-code, bug and quality checker for Python.
- **Terraform**: An infrastructure as code software tool for building, changing, and versioning infrastructure safely and efficiently.

### 5.2 Appendix B: References

- [IEEE Std 830-1998](https://ieeexplore.ieee.org/document/720574)
- [AAI-540 Course Materials and Project Requirements](#)
