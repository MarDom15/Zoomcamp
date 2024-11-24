#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# In[10]:


# Load the dataset
data = pd.read_csv(r'C:\Users\marti\Desktop\Project1_DataZoomcamp\Zoomcamp\Project_1\Data_kredit.csv')   # Replace with your dataset path

# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# If missing values are found, display a warning
if data.isnull().sum().any():
    print("\nWarning: The dataset contains missing values!")
else:
    print("\nNo missing values detected.")


# In[11]:


# Display the first few rows of the dataset
print("\nPreview of the first few rows of the dataset:")
print(data.head())


# In[13]:


# Convert "ja"/"nein" to numerical values in the 'creditRisk' column
if "CreditRisk" in data.columns:  # Check if the column exists
    print("\nEncoding 'CreditRisk' column (ja/nein):")
    data["CreditRisk"] = data["CreditRisk"].map({"ja": 1, "nein": 0})
    print(data["CreditRisk"].value_counts())
else:
    print("\nColumn 'CreditRisk' not found. Skipping encoding.")


# In[16]:


# Separate features (X) and labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[17]:


# Convert X to a pandas DataFrame for compatibility
X = pd.DataFrame(X, columns=data.columns[:-1])  # Keep original column names


# In[18]:


# Check for categorical columns in features
print("\nChecking for categorical columns in X:")
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Categorical columns: {categorical_cols.tolist()}")


# In[19]:


# Encode categorical features
if len(categorical_cols) > 0:
    print("\nEncoding categorical features...")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


# In[20]:


# Verify the preprocessing
print("\nPreprocessed features (X):")
print(X.head())
print("\nTarget variable (y):")
print(y.head())


# In[22]:


# Split the data into training (60%), testing (20%), and evaluation (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display dataset sizes
print(f"\nDataset sizes:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")
print(f"Evaluation: {X_eval.shape[0]} samples")


# In[23]:


# List of models to train and evaluate
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Naive Bayes": GaussianNB()
}


# In[24]:


# Initialize a dictionary to store the results of each model
results = {}


# In[25]:


# Loop through each model
for name, model in models.items():
    print(f"\nTraining the model: {name}")
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # If the model supports probability predictions, calculate ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    else:
        roc_auc = np.nan  # Not applicable for models like SVM without probabilistic kernels
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # Store the results
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc
    }
# Display the classification report
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


# In[26]:


# Display a summary of model performances
print("\nPerformance Summary:")
performance_df = pd.DataFrame(results).T
print(performance_df)


# In[28]:


# Identify the best model based on F1-Score
best_model_name = performance_df["F1-Score"].idxmax()
best_model_score = performance_df["F1-Score"].max()
best_model = models[best_model_name]

# Print details about the best model and the reason for its selection
print(f"\nThe best model selected is: {best_model_name}")
print(f"Reason: {best_model_name} achieved the highest F1-Score of {best_model_score:.2f} among all evaluated models.")
print(f"This indicates that {best_model_name} strikes the best balance between precision and recall, "
      "making it the most reliable for classification in this scenario.")


# In[29]:


# Save only the best model
joblib.dump(best_model, 'best_model.pkl')
print(f"\nThe model {best_model_name} has been saved as 'best_model.pkl'")


# In[30]:


# Evaluate the best model on the evaluation set
print("\nEvaluating the best model on the evaluation set:")
y_eval_pred = best_model.predict(X_eval)
eval_accuracy = accuracy_score(y_eval, y_eval_pred)
eval_f1 = f1_score(y_eval, y_eval_pred, average="weighted")
print(f"Accuracy on the evaluation set: {eval_accuracy:.2f}")
print(f"F1-Score on the evaluation set: {eval_f1:.2f}")


# In[ ]:




