# Predictive Maintenance - Machine Learning Project

This project focuses on **predictive maintenance** by analyzing equipment sensor data to predict potential failures.  
The goal is to **classify** failure types or detect if a failure will occur, using machine learning techniques.

> **Status:** Work in progress 

##  Project Structure
.
├── src/
│ └── preprocess.py # Preprocessing pipeline for data cleaning & feature transformation
├── tests/
│ └── test_preprocess.py # Unit tests for preprocessing
├── data/
│ └── predictive_maintenance.csv # Dataset (not tracked in Git)
├── README.md
├── requirements.txt
└── .gitignore


##  Current Features
- Data validation & error handling
- Missing value handling (numeric → median, categorical → mode)
- Train-test split with optional stratification
- Scikit-learn `Pipeline` for numerical scaling and categorical encoding
- Unit tests with `pytest`

##  Dataset
The dataset is taken from [Kaggle: Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification).  
It contains **10,000 entries** with sensor readings such as:
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Failure Type and Target

##  Installation
```bash
# Clone repository
git clone https://github.com/Livanturk/Machine-Predictive-Maintenance.git
cd <repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
