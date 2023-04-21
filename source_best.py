import numpy as np
import pandas as pd
import arff
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from autosklearn.classification import AutoSklearnClassifier

# Load the dataset
with open("phpgNaXZe.arff", "r") as f:
    dataset = arff.load(f)
data = np.array(dataset["data"])
df = pd.DataFrame(data, columns=[attr[0] for attr in dataset["attributes"]])

# Preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.astype(int) - 1
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create an AutoML classifier with only neural networks
automl = AutoSklearnClassifier(
    time_left_for_this_task=600,
    ensemble_size=0,
    include_estimators=["mlp", "adaboost-mlp"],
    exclude_estimators=["libsvm_svc", "sgd", "pa", "passive_aggressive"],
)

# Fit the AutoML classifier to the training data
automl.fit(X_train, y_train)

# Evaluate the AutoML classifier on the test data
y_pred = automl.predict(X_test)
y_pred_prob = automl.predict_proba(X_test)[:, 1]

# Analyze performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Confusion Matrix:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_prob)
print("AUC Score:", auc_score)
