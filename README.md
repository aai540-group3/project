[![Deploy HF Space: OpenAI TTS](https://github.com/aai540-group3/project/actions/workflows/deploy-tts-space.yml/badge.svg)](https://github.com/aai540-group3/project/actions/workflows/deploy-tts-space.yml)

# Final Project

* **Due:** Oct 21 by 11:59pm
* **Points:** 320
* **Submitting:** a text entry box, a website url, a media recording, or a file upload
* **Allowed Attempts:** 3

## Description

Throughout AAI-540, you will learn and practice the framework, methodology, and skills for machine learning operations (MLOps). For the course project, you will form small groups (2-3 members) and apply your knowledge of MLOps in a real-world ML project scenario. Over the span of the course, your team will iteratively design and develop the key components for a production-ready machine learning system. In each module, incorporate MLOps concepts into your team project plan and practice team workflows to iteratively design and build your machine learning system. In Module 7, your team will deliver a final ML System Design Document and codebase and demonstrate the operation of your ML system.

#### **Project Scenario**

In the first module, beginning with the project data sets, you will consider a problem space, identify potential ML problems for your project scenario, and form small groups around ML project ideas. Your group will research a hypothetical company and business goal to serve as the impetus for the machine learning system product solution that your team will design and build. You will frame a problem that a machine learning system can solve. When selecting your problem, make sure your problem space meets the requirements of a machine learning system (see Module 1 presentations). Additionally, when selecting your data source, make sure that there is enough data to solve your problem (as a rule of thumb, the minimum dataset should be 3-5 files, with 2 files having at least 10,000 records in a raw dataset). If you are doing a classification problem, you should have a minimum of 10,000 labeled records per class that you are predicting.

#### **Project Documents**

1. Download the AAI-540 [Design Document Template](https://sandiego.instructure.com/courses/14123/files/1940576?wrap=1).

2. Copy and share the AAI-540 [Project Tracker](https://docs.google.com/document/d/1ar3DZ6YA_bmo9P4Jq9UIJs788deCf1ReQ0HbyX1VgrU/template/preview).

#### **Project Data Sets**

* [Kaggle](https://www.kaggle.com/datasets)
* [Yelp](https://www.yelp.com/dataset)
* [AWS Open Data](https://registry.opendata.aws/)
* [Data.gov](https://data.gov/)
* [US Census Data](https://www.census.gov/data/tables.html)
* [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)

#### **Project Deliverables**

Refer to the Requirements section and detailed descriptions of the deliverables due further below. 

1. Team Document - Written report "ML System Design Document"
2. Team Video + Script/Outline - video presentation with screencast demonstration of "ML System Operation Validation" 
3. Team Code - GitHub repository with the complete codebase for your ML system

#### **Project Timeline**

In each module, you will learn new skills and knowledge to include in your team project. The recommended steps for each module are summarized in the module and [Project Tracker](https://docs.google.com/document/d/1ar3DZ6YA_bmo9P4Jq9UIJs788deCf1ReQ0HbyX1VgrU/template/preview). In Module 2, you will be prompted to create a team Asana board for planning your project and tracking team workflows. In each module, locate and read the Team Project Steps for the module week; these steps may be either required or recommended based on module topics. Note that in Module 2 and Module 4, your team will complete Team Project Check-Ins as graded assignments.

Familiarize yourself with the project steps and deliverables of each module and update your Asana project board. Keep pace with the project; if you complete each piece week by week, the project will not be difficult. If you wait until the last week, the project will be tough to complete. You can proceed at your own pace, and you are welcome to go back and redo parts of the project.

#### **Project Examples**

An example of a student teamwork and deliverables from AAI-540 Spring 2024 is provided in Module 2: 

* Team Project Presentation: "Smart Meter Energy Prediction" 

---

## Requirements

Divide the work equally between the team members for the following steps, and everyone needs to work on at least one portion of the project in SageMaker. You are expected to write high-quality, efficient, and readable code in Python. This project requires that you and your team create a technical design document, create a project plan and tickets on Asana, implement a full machine learning system and push your code to a GitHub repository, and validate the operability of your ML system in a brief video demonstration.  Your GitHub repository should be clean and professional. Your data files should be stored in GitHub or S3.

There are three group deliverables due at the end of Module 7. Please read the requirements for each deliverable thoroughly, discuss team workflows for collaborative development, and plan time each week to progress toward the final deliverables.

#### **Participation**

This is designed as a *team* project and successful delivery will require *teamwork.* While students personal schedules may differ, equitable contributions, ongoing communication, and team project management is *required.* Team members who do not contribute to the Team Project iterations and outputs may receive point deductions. To support individual accountability and participation, in AAI-540 you will:  

1. Post weekly Project Updates using a [teamwork tracker](https://docs.google.com/document/d/1ar3DZ6YA_bmo9P4Jq9UIJs788deCf1ReQ0HbyX1VgrU/template/preview).
2. Complete Peer Evaluations in Module 7 using this [Peer Evaluation Form](https://sandiego.instructure.com/courses/14123/files/1940545?wrap=1). 

#### **Lateness Policy**

To avoid late penalty deductions, assignments should be submitted on or prior to the due date listed in the course. For each day after the deadline that an assignment is turned in late, its point value will be reduced by 10% per day. Work submitted after 48 hours/two days past the original due date will not be accepted and will receive a zero unless the matter is discussed with the instructor in advance.

---

### **Deliverable #1: Design Document**

#### ML System Design

1. Finalize the [ML System Design Document](https://sandiego.instructure.com/courses/14123/files/1940576?wrap=1) that you have drafted throughout the course for final submission to project stakeholders.
2. In your final document, make sure you include the following:
    * Include a clearly defined problem statement. This should be one to two paragraphs.
    * Include a clear description of how you will measure the impact of your work; this should tie directly to the goals. This should be one to two paragraphs.
    * Complete the security checklist and describe any risks surrounding sensitive data, bias, and ethical concerns.
    * **Solution Overview.** Describe your implementation of the solution. Each section should be two to three paragraphs and should describe your findings from the code in your GitHub repository. Your answers should be detailed and explain your rationale for the decisions you have made.
        * Data Sources
        * Data Engineering
        * Feature Engineering
        * Model Training & Evaluation
        * Model Deployment
        * Model Monitoring
        * CI/CD

**Final Format:**

The ML Design Document is for a technical audience and must be written in a clear, organized fashion. Refer to the scoring rubric for success criteria and guidance. 

---

### **Deliverable #2: Video Demonstration**

#### ML System Operation Validation

1. Implement the project using AWS SageMaker.
2. Validate ML system operation.
3. Demonstration should briefly introduce the business use case, architecture diagrams, and a demonstration of the components in action.
4. Future improvements and any difficult issues that came up during project development, or any risks that might effect future development (scalability, technical risks, or ethical risks). 
5. Plan and record a 10-15 minute video that demonstrates the following:  

    * feature store and feature groups
    * infrastructure monitoring dashboards for your system
    * any model or data monitoring reports
    * CI/CD DAG in a successful and failed state
    * model registry
    * outputs of batch inference job or endpoint invocation

**Video Guidance:**

* Record/screencast your screen and provide voice narration of the above content requirements. Consider using a recording software, such as Screencast-O-Matic or Zoom.
* Explore script or transcript capabilities in the video editor of your team's choice.
* Ensure that the sound quality of your video is good and each member presents an equal portion of the presentation. Export the video file to an **mp4 format**. 
* View the Recording Video Presentation and Submission Guidelines in the **Course Resources.**

**Final Format:**

For this deliverable, your team will submit: 

1. A 10-15 minute video in .mp4 format with screencast and audio narration of required items.
2. An outline, slide notes, or a transcript of the video demonstration.

---

### **Deliverable #3: Code**

#### Codebase GitHub Repository

The final presentation and submission of your codebase should support your ML Design Document and reflect teamwork. 

1. Include a link to your team’s GitHub repository in your design document. 
2. Your GitHub repository should reflect the following **eight** requirements: 
    * **Method**
        1. All of your code should be stored in GitHub in a clean and professional manner. Notebooks should be stored in .ipynb format.
        2. Your code should be clean, have useful comments, and only include code that builds towards the project goal.
        3. Your data should be stored in S3 and documented in your GitHub repository
        4. Any graphics, such as charts/graphs that help explain your data, should be included in your .ipynb files.
    * **ML Design**
        1. The codebase should be comprehensive and complete as an ML system codebase.
        2. The codebase and design document should be mutually reinforcing, reflecting the parallel effort and scope of the ML system. 
    * **Teamwork**
        1. All team members should contribute to the GitHub repository.
        2. Commit history will be available to the instructor for review.

---

## Submission

* **File naming:** Use a standardized file naming convention for each of the deliverables. Example: **Final-Project_Team-1_Deliverable-1.pdf.** 
* **Team delegate:** Determine a project delegate to submit the final deliverable(s). Only one member of your team should submit for the team.
* **Please note:** Since this assignment requires the submission of more than one item, files must be submitted *at the same time* (submissions overwrite rather than add, avoid submitting deliverables one at a time). 
* **Teamwork**: You will submit the Peer Evaluation form individually using the separate assignment link in Module 7. Consult the syllabus for grading weights of the team project and peer evaluations. Team members may not get the same grade on the Final Team Project, depending on each team member's level of contribution.
* **Turnitin:** This assignment has [Turnitin](https://help.turnitin.com/integrity/student/canvas/assignments/submitting-an-assignment.htm) enabled for document submissions which means that your instructor will obtain a Similarity Report that identifies specific parts of your writing that may indicate a high level of matching to external content. You are strongly encouraged to review your work without penalty by activating the [Draft Coach extension in your Google Docs](https://help.turnitin.com/integrity/student/draft-coach/using-draft-coach.htm) prior to submitting your work for final grading.

---

## Rubric

| Criteria | Meets or Exceeds Expectations (Points) | Approaches Expectations (Points) | Below Expectations (Points) | Inadequate Attempt (Points) | Non-Performance (Points) |
|---|---|---|---|---|---|
| **ML Design Document (100 pts)** | An integrated and cohesive ML Design that connects a deep understanding of course knowledge and project requirements to a thoughtfully researched context and goal. Details reflect a dedicated and iterative process. Problem statement is clearly stated and is a ML problem. The design document is complete and communicates a depth and understanding of project ramifications and possible issues that extend beyond the assignment requirements. The solution details are documented clearly and the question(s) are technically adept and reflect industry-level analysis and implementation. Outlined requirements include details that are further reinforced and consistent with project demonstration and codebase. The document is effective, clear, free of all grammatical and spelling errors, and presentable to a technical audience in its current form. (100) | A somewhat cohesive ML Design that addresses project requirements, but may lack depth of context or understanding of the technical audience and needs. ML Problem is identified and described. Project objectives are met, and almost all sections of the technical design document are replete. Solution details are documented, and important question(s) are technically addressed in the report. Outlined requirements include details that are further reinforced and consistent with project demonstration and codebase.  The document is effective but contains a few minor grammatical and spelling errors. (90) | ML problem is not clearly identified and described. Project objectives are mostly met, but some sections of the technical design document are incomplete.  Solution details are reasonably documented, but technical aspects and details are absent. Outlined requirements include details that are not reinforced and consistent with project demonstration and codebase. The document poses challenges to a  technical audience requiring supporting detail. The document is not presentable to a professional technical audience because it is not clearly written and/or contains grammatical and spelling errors that detract from its effectiveness. (82) | Submission does not meet graduate level standards. (70) | Non-performance. (0) |
| **ML Demonstration (80 pts)** | Demonstration is not over 10 minutes in length, includes an outline or transcript, and is presentable to business stakeholders in its current form. Functional details are contextualized to the outlined business problem and presented to a stakeholder audience as a defensible solution design. Screencast validates the operability of the ML system and demonstrates all listed ML system elements. Visual and narrated are planned or scripted, clear, and well-defined. Audio and video are edited to remove glitches and outtakes and represent a professional team output. Demonstration of the ML system components is equally and professionally presented by the entire team. (80) | Demonstration is between 10 to 15 minutes in length but does not demonstrate or include an outline or script. Presentation of the business context is lacking. Planning, coordination, and/or editing would make this demonstration a more polished presentation for business stakeholders. Screencast validates the operability of the ML system and most of the listed ML system elements. Visual and audio are reasonably clear, but contain a few glitches or outtakes that could have been edited for a professional audience.  Coordination and planning would improve overall communication of the team effort; components might not be presented equally or are not completely aligned with other project components. (72) | Presentation exceeds 15 minute maximum and requires coordination and planning to only present required details. Business context and audience is not represented effectively. Screencast is an incomplete validation of the operability of the ML system and missing several key  ML system elements. Visual and audio may lack clarity or plausibility; or there are so many glitches and outtakes that it is ineffective as a professional communication. Overall, the demonstration of the ML system components lacks professional credibility, and/or is not equally presented by the entire team. (65.6) | Submission does not meet graduate level standards. (56) | Non-performance. (0) |
| **ML Code (140 pts)** | The codebase demonstrates all 8 of the eight listed requirements for Methods, ML Design, and Teamwork. (140) | The codebase demonstrates 6-7 of the eight listed requirements for Method, ML Design, and Teamwork. (126) | The codebase demonstrates 4-5 of the eight listed requirements for Method, ML Design, and Teamwork. (114.8) | Codebase demonstrates 3 or fewer listed requirements for Method, ML Design, and Teamwork. (98) | Non-performance. (0) |

**Total Points: 320** 
