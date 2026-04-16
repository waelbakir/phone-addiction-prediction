# Predicting Digital Dependency: A Data Mining Approach to Smartphone Addiction 

## Project Overview
In the modern digital era, smartphone addiction (nomophobia) has emerged as a significant psychological and behavioral concern. Excessive screen time, particularly on social media and gaming, is increasingly linked to poor sleep quality, elevated stress levels, and declining academic or professional performance. 

The goal of this project is to use data mining and machine learning techniques to identify the hidden patterns of smartphone usage that lead to addiction. By predicting an individual's addiction status based on their passive usage metrics, we aim to uncover insights that could power proactive digital well-being interventions.

## The Dataset
The project utilizes a dataset of 7,500 anonymized user records. 
* **Features:** 15 parameters including daily screen time, social media hours, gaming hours, sleep hours, daily notifications, app opens, stress levels, and academic/work impact.
* **Target Variable:** `addicted_label` (Binary: 0 = Not Addicted, 1 = Addicted).
* **Note:** The `addiction_level` feature is dropped during modeling to prevent data leakage.

## Project Methodology
This project strictly follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:
1. **Business Understanding:** Defining the problem and objectives.
2. **Data Understanding:** Exploratory Data Analysis (EDA), covariance, and correlation analysis.
3. **Data Preparation:** Data cleaning, handling categorical variables (One-Hot Encoding/Mapping), and feature scaling.
4. **Modeling:** Training a **Logistic Regression** model for binary classification.
5. **Evaluation:** Assessing the model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
6. **Deployment:** Structuring the repository for reproducibility.

## Repository Structure
```text
phone-addiction-prediction/
│
├── data/                   # Contains the dataset (e.g., transaction_id.csv)
├── docs/                   # Project reports, CRISP-DM plan, and presentation slides
├── notebooks/              # Jupyter notebooks containing EDA, data cleaning, and ML models
├── src/                    # Standalone Python scripts (if applicable)
├── requirements.txt        # List of Python dependencies required to run the code
└── README.md               # Project documentation
Setup & Installation
To run this project locally, follow these steps:
Clone the repository:
code
Bash
git clone https://github.com/waelbakir/phone-addiction-prediction.git
cd phone-addiction-prediction
Install the required dependencies:
Make sure you have Python installed, then run:
code
Bash
pip install -r requirements.txt
Run the Jupyter Notebook:
code
Bash
jupyter notebook
Navigate to the notebooks/ folder and open addiction_analysis.ipynb.
Contributors
Wael Bakir
Bakhom hany
