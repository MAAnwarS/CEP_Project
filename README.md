## 🛡️ Phishing URL Detection Using Machine Learning
This project implements a complete end-to-end machine learning pipeline for detecting phishing URLs, powered by a well-structured ETL workflow, MLflow experiment tracking, Dockerization, and cloud deployment on AWS EC2. It uses a binary classification approach based on 30+ handcrafted features extracted from URLs.

## 📁 Project Structure
This repository follows a modular and production-ready structure with components like:

**data_ingestion/** - Fetching and storing data from sources like MongoDB Atlas

**data_validation/** - Ensuring data quality and schema checks

**data_transformation/** - Preparing features for ML models

**model_trainer/** - Training and hyperparameter tuning with MLflow

**batch_prediction/**- Inference pipeline for real-time or batch predictions


## 🧪 Features
**🔁 ETL Pipeline** built using Python and MongoDB Atlas

**🧹 Data Validation** & Transformation with schema enforcement

**🤖 Model Training** using algorithms with hyperparameter tuning

**📊 MLflow Experiment** Tracking (with Dagshub as remote store)

**🛠️ Logging and Exception Handling** throughout the code

## 🚀 End-to-End Flow
1. Set up Environment & Project Structure

2. Initialize GitHub Repo & VS Code Integration

3. Create setup.py for Packaging

4. Implement Logging & Exception Handling

5. Design ETL Pipeline and Integrate with MongoDB Atlas

6. Build Data Ingestion, Validation & Transformation modules

7. Train Models with Evaluation and Hyperparameter Tuning

8. Track Experiments using MLflow with Remote Dagshub

9. Create Batch Prediction Pipeline


## ⚙️ Tech Stack
- Python

- MongoDB Atlas

- Scikit-learn

- MLflow + Dagshub


## 🧰 Setup Instructions

#### Clone the repo
git clone https://github.com/MAAnwarS/CEP_Project.git
cd network_security

#### Create a virtual environment
python -m venv venv
source venv/bin/activate  # for Unix
venv\Scripts\activate     # for Windows

#### Install dependencies
pip install -r requirements.txt

#### Run the pipeline (example command)
python main.py


## 📊 MLflow UI (Local or Remote)
mlflow ui



