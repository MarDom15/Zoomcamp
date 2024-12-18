# 🚀 Supervised Classification Application

## 📚 Introduction
In the banking and financial sector, data management and its exploitation through artificial intelligence (AI) models are strategic priorities. This project proposes a supervised classification based on multiple machine learning models to solve critical problems such as:

- **Risk Analysis**
- **Fraud Detection**
- **Customer Segmentation**
- **Portfolio Optimization**
- **Regulatory Compliance**

With an intuitive interface and full automation, this project aims to maximize added value in this highly competitive field. 🌟

---

## 🎯 Project Objectives

1. **Development of a classification pipeline**
   - Implementation of several supervised models:
     - KNN
     - Random Forest
     - SVM
     - Logistic Regression
     - Naive Bayes
   - Calculation and display of performance metrics:
     - Accuracy
     - F1-Score
     - Precision
     - Recall
     - ROC AUC
     - Confusion Matrix
   - Automatic selection of the best model. 🏆

2. **Saving the optimal model**
   - Exporting the best-performing model as a `.pkl` file.

3. **Development of a user interface via Streamlit**
   - Submitting data for predictions.
   - Visualizing model performances.

4. **Containerization with Docker**
   - Portable and simple deployment.

---

## 📦 Deliverables

- Functional Python scripts:
  - Main program for training and evaluating models.
  - Automatic saving of the best model.
- Interactive Streamlit application.
  - Direct access to the application: [Streamlit App](https://project1-9hv5popj3psoy7al77y8s7.streamlit.app/)
- Docker image ready for deployment.
- Comparative report of model performances. 📊

---

## 🛠️ Project Steps

1. **Data Analysis and Preparation**
   - Data cleaning.
   - Separation of features (X) and labels (y).
   - Splitting data into training and testing sets. 🧹

2. **Model Implementation**
   - Creation of supervised models:
     - K-Nearest Neighbors (KNN)
     - Random Forest
     - Support Vector Machine (SVM)
     - Logistic Regression
     - Naive Bayes
   - Comparing performances using defined metrics. ⚙️

3. **Automation of Best Model Selection**
   - Computing metrics for each model.
   - Automatically saving the model with the best F1-Score. 💾

4. **Development of Streamlit Interface**
   - Submitting data to obtain predictions.
   - Visualizing performances (table and confusion matrix).

5. **Containerization with Docker**
   - Creating a Dockerfile.
   - Generating a Docker image. 🐳

6. **Documentation**
   - Performance explanatory report.
   - User documentation for the Streamlit application and Docker container. 📝

---

## ✅ Success Criteria

1. **Minimum Accuracy:**
   - F1-Score > 85% on test data.
2. **Intuitive User Interface:**
   - Easy data submission.
   - Clear predictions.
3. **Successful Containerization:**
   - Seamless functionality within a Docker container.
4. **Performance Report:**
   - Clear model comparison.

---

## 👥 Target Audience

- **Data Analysts:** To analyze model performances.
- **Developers:** To quickly integrate predictions into their applications.
- **End Users:** To interact through a simple interface.

---

## 🔧 Technologies Used

- **Python** (scikit-learn, joblib, streamlit, matplotlib, seaborn, pandas, numpy)
- **Docker** (for containerization)
- **Streamlit** (for the user interface)

---

## 🌟 Final Objective
To create a complete and portable supervised classification solution that:

1. Identifies the best-performing machine learning model for specific data.
2. Provides an intuitive interface for real-time predictions.
3. Delivers a containerized application easily deployable in any environment. 🌐
