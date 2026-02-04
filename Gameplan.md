Based on the **Project Assessment Criteria** and the **Predicting Bike Sharing Demand** dataset, here is the full execution plan. This plan is designed to hit every single grading requirement, specifically the 30% allocated to **Data Augmentation** and **Dashboarding**.

### **I. Team Roles (The "Divide and Conquer" Strategy)**

Assign these roles immediately. Do not overlap tasks or you will waste time.

- **MEMBER A: The Data Architect (Focus: Augmentation & SQL)**
- **Goal:** Secure the 15% Data Augmentation grade and the 25% Cleaning grade.
- **Key Deliverable:** `train_augmented.csv` (The final clean dataset used by everyone) and the `augmentation_queries.sql` file.

- **MEMBER B: The Modeler (Focus: Predictions & Kaggle)**
- **Goal:** Secure the 20% Modelling grade and the Kaggle Submission requirement.
- **Key Deliverable:** The Kaggle Submission Screenshot and the Model Analysis section of the report.

- **MEMBER C: The Visualizer (Focus: Dashboard & Story)**
- **Goal:** Secure the 15% Dashboard grade and the 15% Presentation grade.
- **Key Deliverable:** The Interactive Dashboard (Python script) and Presentation Slides.

---

### **II. Detailed Weekly Instructions**

#### **Phase 1: Setup & Augmentation (Days 1–5)**

**Goal:** Create a "Super Dataset" that has more information than the original Kaggle file.

- **MEMBER A (Architect):**

1. **Find External Data:** Download a 2011–2012 **Holiday Calendar** (CSV) and **Weather History** (Precipitation in mm) for Washington D.C.
2. **SQL Augmentation:** Load the Kaggle data and your external data into a temporary SQL database (or use pandasql in Python). Write a query to `LEFT JOIN` them on the `Date` column.

3. **Save:** Export the result as `train_augmented.csv` and share it with the team.
4. **Documentation:** Save your SQL query code into a text file named `data_augmentation.sql` (Required for submission).

- **MEMBER B (Modeler):**

1. **Baseline Model:** Take the _original_ raw data and run a simple Random Forest. Check the RMSLE score. This is your "benchmark" to prove your final model is better later.
2. **Metric Research:** Understand exactly how "RMSLE" (Root Mean Squared Log Error) works, as this is how Kaggle judges you.

- **MEMBER C (Visualizer):**

1. **Initial EDA:** Use the raw data to plot **Average Rentals per Hour**. You should see two spikes (8 AM and 5 PM).
2. **Hypothesis Generation:** Write down 3 hypotheses for the report (e.g., "Heavy rain reduces rentals by >50%").

---

#### **Phase 2: Feature Engineering & Analysis (Days 6–10)**

**Goal:** Transform raw data into "Signal" for the model.

- **MEMBER A (Architect):**

1. **Clean the Mess:** The external weather data might have "N/A" for some days. Decide how to fill them (e.g., fill with 0 or the previous day's value). Document this decision in the report .

2. **Check Outliers:** Identify hours with 0 rentals. Is the system down? Or is it 3 AM?

- **MEMBER B (Modeler):**

1. **Feature Creation:** Create specific columns that help the model:

- `hour` (0-23)
- `day_of_week` (Mon-Sun)
- `is_rush_hour` (If hour is 7-9 or 17-19 on a weekday = 1, else 0).

2. **Encoding:** Convert categorical variables (like Season 1,2,3,4) into "One-Hot Encoded" columns if using Linear Regression.

- **MEMBER C (Visualizer):**

1. **Validate Features:** Plot `is_rush_hour` vs `rentals`. If the chart shows a massive difference, the feature is good.
2. **Dashboard Prototype:** Start building the layout in Streamlit or Dash. Create the sidebar with empty filters for now.

---

#### **Phase 3: Building & Tuning (Days 11–15)**

**Goal:** Get the high score and build the tool.

- **MEMBER B (Modeler) - CRITICAL:**

1. **Train Final Models:** Train an **XGBoost** or **Random Forest** regressor on the `train_augmented.csv`.
2. **Submit to Kaggle:** Generate the `submission.csv` file. Upload it to Kaggle.
3. **Screenshot:** **Take a screenshot of the "Success" page and the Leaderboard**. This is mandatory.

- **MEMBER C (Visualizer) - CRITICAL:**

1.  **Build Filters:** Implement the required interactivity:

- **Date Range Slider:** Let user pick "Jan 2011 to Mar 2011".
- **Weather Filter:** Checkbox for "Show only Rainy Days".

2. **Add Predictions:** Add a button that runs Member B's model on new inputs (e.g., "Predict demand for next Tuesday at 5 PM").

- **MEMBER A (Architect):**

1. **Report Writing:** Begin writing the "Data Cleaning" and "Augmentation" chapters of the Jupyter Notebook. Explain _why_ you chose the external data you did.

---

#### **Phase 4: Assembly & Polish (Days 16–20)**

**Goal:** Package it all up for the grade.

- **ALL HANDS:**
- **Presentation Practice:** 10 Minutes total. Member C demos the dashboard live. Member B explains the model. Member A explains the data.

- **FINAL CHECKLIST (Don't miss these):**
- [ ] **Jupyter Notebook:** With comments explaining every step.

- [ ] **Dashboard Script:** The `.py` file.

- [ ] **Augmentation Code:** The SQL queries used.

- [ ] **Kaggle Proof:** The screenshots embedded in the report.

- [ ] **Slides:** Saved as PDF.

- [ ] **Zipped:** Named `ACADA6_GroupNo_Name1_Name2_Name3.zip`.

### **III. Immediate Next Step (For Tonight)**

**Member A** needs to run the **Python Augmentation Script** I provided in the previous turn to generate the `train_augmented.csv`. This unlocks the work for everyone else.

**Would you like me to write the "Skeleton Code" for the Dashboard (Member C's task) so they have a template to start with?**
