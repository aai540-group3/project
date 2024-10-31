# Software Requirements Specification (SRS)

## Document Version

1.0

## Date

September 30, 2024

## Project Name

Machine Learning System for Predicting Hospital Readmissions

---

## Table of Contents

- [Software Requirements Specification (SRS)](#software-specification-srs)
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
      - [FR-1: Problem Definition](#fr-1-problem-definition)
      - [FR-2: Impact Measurement](#fr-2-impact-measurement)
      - [FR-3: Security and Ethical Compliance](#fr-3-security-and-ethical-compliance)
      - [FR-4: Data Sources](#fr-4-data-sources)
      - [FR-5: Data Engineering](#fr-5-data-engineering)
      - [FR-6: Feature Engineering](#fr-6-feature-engineering)
      - [FR-7: Model Training and Evaluation](#fr-7-model-training-and-evaluation)
      - [FR-8: Model Deployment](#fr-8-model-deployment)
      - [FR-9: Model Monitoring](#fr-9-model-monitoring)
      - [FR-10: CI/CD Processes](#fr-10-cicd-processes)
      - [FR-11: Documentation](#fr-11-documentation)
      - [FR-12: Video Demonstration](#fr-12-video-demonstration)
      - [FR-13: Video Outline or Transcript](#fr-13-video-outline-or-transcript)
      - [FR-14: Codebase Requirements](#fr-14-codebase-requirements)
    - [3.2 Non-functional Requirements](#32-non-functional-requirements)
      - [NFR-1: Performance](#nfr-1-performance)
      - [NFR-2: Scalability](#nfr-2-scalability)
      - [NFR-3: Reliability](#nfr-3-reliability)
      - [NFR-4: Maintainability](#nfr-4-maintainability)
      - [NFR-5: Usability](#nfr-5-usability)
      - [NFR-6: Security](#nfr-6-security)
      - [NFR-7: Compliance](#nfr-7-compliance)
      - [NFR-8: Team Collaboration Efficiency](#nfr-8-team-collaboration-efficiency)
    - [3.3 External Interface Requirements](#33-external-interface-requirements)
      - [EIR-1: User Interfaces](#eir-1-user-interfaces)
      - [EIR-2: Hardware Interfaces](#eir-2-hardware-interfaces)
      - [EIR-3: Software Interfaces](#eir-3-software-interfaces)
      - [EIR-4: Communications Interfaces](#eir-4-communications-interfaces)
  - [4. Project Deliverables](#4-project-deliverables)
    - [4.1 Deliverables Overview](#41-deliverables-overview)
    - [4.2 Deliverables List](#42-deliverables-list)
      - [D-01: ML System Design Document](#d-01-ml-system-design-document)
      - [D-02: Video Demonstration](#d-02-video-demonstration)
      - [D-03: Video Outline or Transcript](#d-03-video-outline-or-transcript)
      - [D-04: Codebase Git Repository](#d-04-codebase-git-repository)
  - [5. Appendices](#5-appendices)
    - [Appendix A: Glossary of Terms](#appendix-a-glossary-of-terms)
    - [Appendix B: References](#appendix-b-references)

---

## 1. Introduction

### 1.1 Purpose

The purpose of this Software Requirements Specification (SRS) document is to define the requirements for developing a machine learning (ML) system that predicts hospital readmissions. This system is intended for a team project as part of the AAI-540 course, focusing on applying machine learning operations (MLOps) practices to design and build a production-ready ML system.

### 1.2 Scope

The ML system aims to predict hospital readmissions by processing and analyzing medical data. The scope includes:

- Defining the ML problem and objectives.
- Measuring the impact of the ML system.
- Ensuring security, ethical considerations, and compliance.
- Developing the ML pipeline, including data ingestion, data engineering, feature engineering, model training, evaluation, deployment, monitoring, and CI/CD processes.
- Delivering required documentation and presentations as per course requirements.

### 1.3 Definitions, Acronyms, and Abbreviations

- **ML**: Machine Learning
- **MLOps**: Machine Learning Operations
- **SRS**: Software Requirements Specification
- **CI/CD**: Continuous Integration/Continuous Deployment
- **API**: Application Programming Interface

### 1.4 References

- AAI-540 Course Materials and Project Requirements
- IEEE Std 830-1998: IEEE Recommended Practice for Software Requirements Specifications

### 1.5 Overview

This SRS document provides an overall description of the ML system, specific functional and non-functional requirements, deliverables, and appendices containing relevant information. It is intended to guide the development team in understanding the system requirements without specifying implementation details.

---

## 2. Overall Description

### 2.1 Product Perspective

The ML system is an independent project developed by a team of students as part of their coursework. It leverages MLOps practices to create a production-ready ML system that addresses the problem of predicting hospital readmissions.

### 2.2 Product Functions

- **Problem Definition**: Define the ML problem of hospital readmission prediction.
- **Impact Measurement**: Establish metrics to measure the impact of the ML system.
- **Security and Ethical Compliance**: Address security, bias, and ethical concerns.
- **Data Ingestion**: Acquire and manage relevant datasets.
- **Data Engineering**: Perform data cleaning and preprocessing.
- **Feature Engineering**: Extract and create meaningful features.
- **Model Training and Evaluation**: Train models and evaluate their performance.
- **Model Deployment**: Deploy models for inference.
- **Model Monitoring**: Monitor model performance over time.
- **CI/CD Processes**: Implement automated integration and deployment pipelines.

### 2.3 User Classes and Characteristics

- **Data Scientists**: Users who develop and validate ML models.
- **ML Engineers**: Users responsible for deploying and maintaining the ML system.
- **Healthcare Stakeholders**: Users who utilize the predictions to make informed decisions.
- **Course Instructors**: Evaluators who assess the project's adherence to requirements.

### 2.4 Operating Environment

- **Development Environment**: Systems capable of supporting ML development tools and languages.
- **Deployment Environment**: Platforms that can host the ML models and serve predictions.
- **Data Storage Environment**: Secure storage solutions for datasets and artifacts.

### 2.5 Design and Implementation Constraints

- The system must comply with the course's project requirements and deadlines.
- The team must collaborate effectively, with equitable contributions.
- The system must be developed using appropriate open-source technologies.
- Data sources must meet the specified size and quality criteria.

### 2.6 Assumptions and Dependencies

- The team has access to necessary computational resources and tools.
- The datasets used are appropriate and available for use without restrictions.
- Team members possess the required skills in ML and MLOps practices.

---

## 3. Specific Requirements

### 3.1 Functional Requirements

#### FR-1: Problem Definition

- **Requirement**: The system shall include a clearly defined problem statement that addresses hospital readmissions as an ML problem.

#### FR-2: Impact Measurement

- **Requirement**: The system shall define clear metrics and methodologies to measure the impact of the ML model on the stated goals.

#### FR-3: Security and Ethical Compliance

- **Requirement**: The system shall complete a security checklist and describe any risks related to sensitive data, bias, and ethical concerns.

#### FR-4: Data Sources

- **Requirement**: The system shall utilize data sources that meet the project's minimum dataset requirements, ensuring sufficient data to build an effective ML model.

#### FR-5: Data Engineering

- **Requirement**: The system shall perform data cleaning, preprocessing, and transformation to prepare the data for modeling.

#### FR-6: Feature Engineering

- **Requirement**: The system shall perform feature engineering to enhance model performance by creating or selecting relevant features.

#### FR-7: Model Training and Evaluation

- **Requirement**: The system shall train one or more ML models and evaluate their performance using appropriate evaluation metrics.

#### FR-8: Model Deployment

- **Requirement**: The system shall deploy the trained ML model for inference, making it accessible for predictions.

#### FR-9: Model Monitoring

- **Requirement**: The system shall implement model monitoring to track performance over time and detect issues such as data drift or degradation.

#### FR-10: CI/CD Processes

- **Requirement**: The system shall implement Continuous Integration and Continuous Deployment pipelines to automate building, testing, and deployment processes.

#### FR-11: Documentation

- **Requirement**: The team shall produce an ML System Design Document that includes detailed descriptions of each component of the system.

#### FR-12: Video Demonstration

- **Requirement**: The team shall prepare a 10-15 minute video that demonstrates the system's operation, including introductions, architecture diagrams, component demonstrations, and discussions of future improvements.

#### FR-13: Video Outline or Transcript

- **Requirement**: The team shall provide an outline, slide notes, or transcript accompanying the video demonstration.

#### FR-14: Codebase Requirements

- **Requirement**: The system's codebase shall be stored in a Git repository in a clean and professional manner, including all necessary code, data documentation, and visualizations.

### 3.2 Non-functional Requirements

#### NFR-1: Performance

- **Requirement**: The system shall provide predictions within acceptable timeframes suitable for practical use.

#### NFR-2: Scalability

- **Requirement**: The system shall be scalable to handle increasing data volumes and user demands without significant performance degradation.

#### NFR-3: Reliability

- **Requirement**: The system shall be reliable, exhibiting minimal downtime and consistent operation.

#### NFR-4: Maintainability

- **Requirement**: The system shall be maintainable, with well-organized and documented code to facilitate updates and troubleshooting.

#### NFR-5: Usability

- **Requirement**: The system shall provide clear interfaces and documentation to ensure ease of use for intended users.

#### NFR-6: Security

- **Requirement**: The system shall ensure the security of data and models, adhering to best practices for data protection and privacy.

#### NFR-7: Compliance

- **Requirement**: The system shall comply with all relevant legal and ethical standards, including data privacy regulations.

#### NFR-8: Team Collaboration Efficiency

- **Requirement**: The team shall demonstrate effective collaboration, using tools to manage workflows, communicate progress, and ensure equitable contributions.

### 3.3 External Interface Requirements

#### EIR-1: User Interfaces

- **Requirement**: The system shall provide appropriate user interfaces (e.g., APIs, dashboards) for users to interact with the ML model and access predictions.

#### EIR-2: Hardware Interfaces

- **Requirement**: The system shall operate on standard hardware used by the development team and end-users, without requiring specialized equipment.

#### EIR-3: Software Interfaces

- **Requirement**: The system shall interface with necessary software platforms and services required for operation, such as data storage solutions and deployment platforms.

#### EIR-4: Communications Interfaces

- **Requirement**: The system shall utilize secure communication protocols for any data transmission to ensure data integrity and confidentiality.

---

## 4. Project Deliverables

### 4.1 Deliverables Overview

The project requires several deliverables that demonstrate the system's design, functionality, and adherence to project requirements.

### 4.2 Deliverables List

#### D-01: ML System Design Document

- **Description**: A comprehensive document that includes:

  - A clearly defined problem statement.
  - A description of how the impact will be measured, tied to project goals.
  - Completion of a security checklist and description of any risks related to sensitive data, bias, and ethical concerns.
  - Detailed descriptions of each component of the solution, including:
    - Data Sources
    - Data Engineering
    - Feature Engineering
    - Model Training & Evaluation
    - Model Deployment
    - Model Monitoring
    - CI/CD Processes

#### D-02: Video Demonstration

- **Description**: A 10-15 minute video presentation that includes:

  - Introduction of the business use case.
  - Architecture diagrams of the system.
  - Demonstration of system components in action.
  - Discussion of future improvements, challenges, and potential risks.
  - Demonstrations of:
    - Feature store and feature groups.
    - Infrastructure monitoring dashboards.
    - Model or data monitoring reports.
    - CI/CD pipeline in a successful and failed state.
    - Model registry.
    - Outputs of batch inference job or endpoint invocation.

#### D-03: Video Outline or Transcript

- **Description**: An outline, slide notes, or transcript that accompanies the video demonstration.

#### D-04: Codebase Git Repository

- **Description**: A Git repository containing the system's codebase, which must:
  - Store all code in a clean and professional manner, with notebooks in `.ipynb` format if used.
  - Include clean code with useful comments, focusing on the project goals.
  - Document data storage solutions (e.g., S3 or repository).
  - Include graphics, such as charts or graphs, within notebooks to explain the data.
  - Reflect a comprehensive and complete ML system codebase.
  - Align with the ML System Design Document.
  - Demonstrate team contributions with a clear commit history.

---

## 5. Appendices

### Appendix A: Glossary of Terms

- **Git**: A distributed version control system for tracking changes in source code.
- **GitHub**: A web-based interface for Git that provides collaboration and code repository hosting.
- **S3**: Amazon Simple Storage Service, an object storage service.
- **CI/CD Pipeline**: A set of processes that automate the integration and deployment of code changes.

### Appendix B: References

- AAI-540 Course Materials and Requirements
- IEEE Std 830-1998: IEEE Recommended Practice for Software Requirements Specifications
- Turnitin Guidelines for Assignment Submission
