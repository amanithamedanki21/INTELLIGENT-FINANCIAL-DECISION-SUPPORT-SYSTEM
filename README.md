# Intelligent Financial Decision Making System

## Project Overview
The Intelligent Financial Decision Making System is a machine learning-based application designed to predict credit card default risk. It analyzes customer financial information and estimates the probability of default.

The system also uses Explainable AI (SHAP) to help users understand the factors influencing model predictions.

## Dataset
- **Source:** UCI Default of Credit Card Clients Dataset
- **Records:** 30,000
- **Original Input Features:** 23
- **Target:** Default Payment Next Month

## Machine Learning Models
1. Logistic Regression
2. Random Forest
3. XGBoost

## Key Features
- Data preprocessing and feature engineering
- Credit card default risk prediction
- Model performance evaluation
- Confusion matrix and ROC curve visualization
- SHAP-based model explainability
- Interactive Streamlit dashboard
- Low, Medium, and High risk categorization

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP
- Streamlit

## Project Structure
```text
Intelligent-Financial-Decision-Making-System/
├── app.py
├── model_training.ipynb
├── requirements.txt
├── README.md
├── models/
├── dataset/
└── images/
```

## Installation and Execution

1. Clone the repository:
   `git clone YOUR_GITHUB_REPOSITORY_URL`

2. Navigate to the project folder:
   `cd Intelligent-Financial-Decision-Making-System`

3. Install dependencies:
   `pip install -r requirements.txt`

4. Run the Streamlit application:
   `streamlit run app.py`

## Explainable AI
SHAP (SHapley Additive exPlanations) is used to explain the influence of different customer features on XGBoost predictions.

## Disclaimer
This project is developed for academic and educational purposes. Predictions should not be treated as professional financial advice.
