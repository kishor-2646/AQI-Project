# 🌬️ Air Quality Index (AQI) Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Machine--Learning-Advanced-blue?style=for-the-badge&logo=machine-learning" alt="ML">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web--App-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
</p>
A Machine Learning project that analyzes air pollution data and predicts the **Air Quality Index (AQI)** using multiple machine learning algorithms.

This project demonstrates the complete **data science workflow**, including:

- Data collection
- Data preprocessing
- Feature engineering
- Model training
- Model comparison
- Data visualization

---



# 📌 Project Overview

Air pollution is one of the biggest environmental challenges today.  
The **Air Quality Index (AQI)** is used to measure how polluted the air is and its impact on human health.
Predicting the Air Quality Index (AQI) is critical for public health and urban planning. This project provides an end-to-end solution—from automated data collection via web scraping to a deployed web interface—allowing users to predict AQI levels based on real-time meteorological parameters.

This project applies different **Machine Learning algorithms** to predict AQI values using environmental and pollution data.

---
## 🖼️ App Preview

| Home Page | Prediction Result | 
| ----- | ----- | 
| ![Home Screenshot](assets/dashboard.png) | ![Result Screenshot](assets/result.jpeg) | 
| *Intuitive Dashboard* | *Instant AQI Analysis* |


# ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## 📁 Project Structure

```text
AQI-Project/
├── assets/                 # App Screenshots & Graphics
├── Data/                   # Raw & Cleaned CSV Datasets
├── Notebooks/              # EDA, Model Training, & Evaluation
├── static/                 # Stylesheets & Brand Assets
├── templates/              # Flask Jinja2 Templates
├── app.py                  # Backend Logic & API Routes
├── model.pkl               # Optimized Saved Model
├── requirements.txt        # Production Dependencies
└── README.md               # Project Documentation
```
---

# 🧠 Machine Learning Models Used

The following algorithms were implemented and compared:

- Linear Regression
- Lasso Regression
- Decision Tree Regressor
- K-Nearest Neighbors (KNN)
- Random Forest Regressor
- XGBoost Regressor
- Artificial Neural Network (ANN)

These models were evaluated to determine which algorithm provides the **best AQI prediction performance**.

---

# 📊 Data Processing Pipeline

### 1️⃣ Data Collection
Air pollution dataset collected from environmental monitoring stations.

### 2️⃣ Data Cleaning
- Handling missing values  
- Removing incorrect records  
- Formatting dataset

### 3️⃣ Feature Engineering
Selecting important pollution parameters affecting AQI.

### 4️⃣ Model Training
Training multiple machine learning models using the dataset.

### 5️⃣ Model Evaluation
Comparing models based on prediction accuracy.

---

### 🧠 Machine Learning Pipeline
Our system follows a robust data science lifecycle to ensure high accuracy and reliability:

```mermaid
graph LR
    A[Data Scraping] --> B[Feature Engineering]
    B --> C[Outlier Handling]
    C --> D[Model Comparison]
    D --> E[Hyperparameter Tuning]
    E --> F[Model Deployment]
    F --> G[Flask Web App]
```


# 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/kishor-2646/AQI-Project.git
```

### 2️⃣ Navigate to project directory

```bash
cd AQI-Project
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python app.py
```
