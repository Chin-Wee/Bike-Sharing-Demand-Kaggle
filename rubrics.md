Here is a detailed summary of the project rubrics based on the provided document, formatted as a Markdown file.

```markdown
# ACADA Module 6: Capstone Project Assessment Rubrics

## Overview

The capstone project is assessed on six distinct components, totaling 100%. [cite_start]The project must be completed in teams of 2-3 members and involves a full analysis pipeline—from problem definition to an interactive dashboard presentation[cite: 61, 76].

---

## Detailed Grading Criteria

### A. Problem Understanding and Set Up (10%)

**Goal:** Assess how the team perceives the problem and architects the solution.

- [cite_start]**Problem Identification:** Clearly identify the type of problem (e.g., Classification, Regression, NLP)[cite: 91].
- **Hypothesis & Workflow:**
  - [cite_start]State clear hypotheses regarding the data and outcomes[cite: 92].
  - [cite_start]Outline the workflow for how the analysis will be carried out[cite: 92].

### B. EDA, Data Cleaning & Feature Engineering (25%)

**Goal:** Assess proficiency in exploring, cleaning, and engineering data for analysis.

#### 1. Exploratory Data Analysis (EDA)

- [cite_start]**Statistical Interrogation:** Use summary statistics to understand the data[cite: 99].
- [cite_start]**Pattern Recognition:** Identify patterns to validate or defeat the initial hypotheses[cite: 100].
- **Visualization:**
  - [cite_start]Use suitable graphs with clear labels to demonstrate insights[cite: 101].
  - [cite_start]Employ plot customizations (color, size) to represent different dimensions[cite: 102].

#### 2. Data Cleaning and Processing

- **Issue Identification & Rectification:** Identify and fix issues that could hinder analysis, including:
  - [cite_start]Missing values[cite: 111].
  - [cite_start]Uneven scales in data[cite: 112].
  - [cite_start]Handling categorical predictors[cite: 113].
  - [cite_start]Presence of outliers[cite: 114].
  - [cite_start]Imbalanced class labels (specifically for classification tasks)[cite: 115].

#### 3. Feature Engineering

- [cite_start]**Relationship Analysis:** Understand how predictors relate to one another (identify extraneous info)[cite: 124].
- [cite_start]**Selection:** Eliminate irrelevant predictors using feature selection techniques[cite: 124].
- [cite_start]**Creation:** Generate new data points by transforming or combining existing columns[cite: 124].

### C. Data Augmentation (15%)

**Goal:** Assess the ability to gather and utilize _external_ data to enhance the analysis.

- [cite_start]**Requirement:** You **must** query a relational database using SQL[cite: 132].
- **Sources:**
  - [cite_start]Google BigQuery public datasets[cite: 133].
  - [cite_start]Web scraping or Web APIs[cite: 134].
- [cite_start]**Outcome:** The dataset must be supplemented with this external data[cite: 68].

### D. Modelling and Model Evaluation (20%)

**Goal:** Assess the ability to experiment with predictive models and evaluate them rigorously.

- [cite_start]**Implementation:** Use the `sklearn` API to generate predictions[cite: 142].
- [cite_start]**Model Selection:** Choose models appropriate for the specific problem type[cite: 142].
- [cite_start]**Evaluation Strategy:** Apply appropriate strategies to test for model **bias and variance**[cite: 143].

### E. Presentation of Results in Dashboard (15%)

**Goal:** Assess the ability to communicate analysis via a user-friendly, interactive tool.

- [cite_start]**Design:** Develop a clear, well-labeled, and user-friendly dashboard[cite: 151, 152].
- [cite_start]**Interactivity:** **Crucial Requirement:** Include elements that allow users to filter or adjust parameters (e.g., time range, category selection, model inputs) to see dynamic changes[cite: 153].
- **Content:** Display key findings from exploration, preprocessing, and modeling. [cite_start]Where applicable, include model outputs/predictions[cite: 152, 153].

### F. Project Presentations (15%)

**Goal:** Assess communication of the end-to-end analytics process in a structured oral presentation.

- [cite_start]**Format:** 5–10 minute oral presentation[cite: 160].
- [cite_start]**Narrative:** Walk through the full process: Problem Definition → Data Collection → Modeling → Evaluation → Insights[cite: 161].
- [cite_start]**Context:** Explain the business goals and relevance of findings[cite: 162].
- [cite_start]**Justification:** Explain _why_ specific methodological choices, models, and metrics were selected[cite: 164].
- [cite_start]**Live Demo:** Use the interactive dashboard as a core part of the presentation[cite: 165].

---

## Report Specifics

_While the report supports the criteria above, it is assessed specifically on:_

1.  [cite_start]**Technical Accuracy:** Correct application of analysis steps[cite: 260].
2.  **Documentation:**
    - Do not just provide code.
    - [cite_start]Add comments stating **assumptions** made[cite: 261].
    - [cite_start]Provide **justification** for every technique attempted[cite: 262].
    - [cite_start]Include an explicit **Introduction** (hypotheses) and **Conclusion** (judgements on the model)[cite: 264].

## Submission Checklist

[cite_start]Ensure the following are included in the final zipped file (`ACADA6_GroupNo_Name1_Name2_Name3.zip`) [cite: 242-251]:

- [ ] Jupyter Notebook Report
- [ ] Kaggle Submission CSV + Screenshots (Success page & Leaderboard)
- [ ] SQL Queries used for augmentation
- [ ] Python Script for the Dashboard
- [ ] Presentation Slides (PDF)
- [ ] Any additional augmentation code/data
```
