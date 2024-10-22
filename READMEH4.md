
Evaluation Metrics



What I Learned in Course 4 of My Machine Learning Training with DataZoomCamp



In this course, we explored various **evaluation metrics** used in machine learning, especially for binary classification models. The practical case we worked on focused on **churn prediction**, which involves predicting customers who are likely to leave a company. Below are the key concepts and methods I learned to evaluate the performance of models in this context:





1. Evaluation Metrics: Session Overview



The goal is to develop a model capable of predicting customer churn, with an initial accuracy result of 80%.

 What does accuracy mean?

Accuracy : is a metric that measures the proportion of correct predictions out of all predictions made by the model. However, it only provides a partial view of model performance, particularly in the context of imbalanced classification problems such as churn prediction.



Are there other metrics to evaluate our binary classification model?

Yes, several other metrics can be used to better understand a binary classification model's performance, especially when accuracy alone is not sufficient due to class imbalance.



2. Accuracy and Dummy Models



Evaluating a model based on different metrics, not just accuracy, is crucial.



 Definition of Accuracy


Scikit-learn provides the `accuracy_score` function, which computes this metric. However, accuracy alone does not provide a complete picture of performance, especially in cases of class imbalance.



 Logistic Regression

Logistic regression optimizes a threshold (typically 0.5) that maximizes accuracy. However, this may not reflect the model’s ability to properly distinguish between customers likely to churn and those who will not. In situations where the non-churn class is the majority, the model can achieve high accuracy simply by predicting "non-churn" for most customers.



 3. Confusion Matrix



The confusion matrix is a tool that helps to better understand model errors, particularly in cases where there is class imbalance. It measures four possible outcomes for binary classification:



- True Negative (TN) : The model predicted "non-churn" and the customer indeed did not leave (correct prediction).

- False Negative (FN) : The model predicted "non-churn," but the customer actually left (incorrect prediction).

-  True Positive (TP) : The model predicted "churn" and the customer indeed left (correct prediction).

- False Positive (FP) : The model predicted "churn," but the customer did not leave (incorrect prediction).



This matrix helps better understand model performance in scenarios where accuracy can be misleading. It accounts for how errors are distributed across the majority and minority classes.



4. Precision and Recall



- Precision : Represents the proportion of correct positive predictions out of all positive predictions made by the model.



- Recall : Measures the proportion of actual churners that were correctly identified by the model.

It answers the question: "What fraction of the churners did the model correctly identify?"



These two metrics are particularly useful in class imbalance contexts, as they provide a better understanding of how well the model performs on the minority class (churn).



5. ROC Curves



The ROC (Receiver Operating Characteristic) curve is a graphical tool used to evaluate the performance of a classification model across all possible thresholds. It plots **sensitivity (recall) against the false positive rate (FPR) for every possible threshold.



- True Positive Rate (TPR): Identical to recall.

- False Positive Rate (FPR):



The ideal model will have an ROC curve close to the upper left corner of the plot, while a random model will follow the diagonal.



6. AUC (Area Under the Curve) 



AUC represents the area under the ROC curve and provides a quantitative measure of model performance. An AUC close to 1 indicates a highly effective model, while an AUC near 0.5 indicates a model barely better than random guessing. A good model generally has an AUC greater than 0.7.



Scikit-learn offers the `roc_auc_score` and `auc` functions to compute this metric.



7. Cross-Validation



Cross-validation is a model evaluation technique that divides the data into multiple parts (or "folds") to reduce the risk of overfitting and provide a more robust assessment.



In k-fold cross-validation, the model is trained on k-1 parts and tested on the remaining part. This process is repeated k times, and the final result is obtained by averaging the performance across all iterations.



8. Summary



- A metric is a function that outputs a single number to evaluate the performance of a model.

- Accuracy: Can be misleading in cases of class imbalance.

- Precision and Recall: More reliable indicators in imbalanced class scenarios.

- ROC Curve and AUC: Graphical and quantitative tools to evaluate performance across thresholds, even in cases of class imbalance.

- Cross-validation: A method to evaluate and fine-tune hyperparameters more reliably.
