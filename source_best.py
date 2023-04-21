import shutil
shutil.rmtree("/tmp/autosklearn_classification_example_tmp", ignore_errors=True)

import autosklearn.classification
from autosklearn.pipeline.components.classification import _classifiers
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
import numpy as np
import pandas as pd
import arff
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pprint import pprint


# Custom AutoSklearnClassifier class
class CustomAutoSklearnClassifier(autosklearn.classification.AutoSklearnClassifier):
    def _get_hyperparameter_search_space(self, dataset_properties=None):
        cs = ConfigurationSpace()

        # Restrict the configuration space to MLP
        allowed_classifiers = ["mlp"]
        classifier_cs = restrict_classifiers(_classifiers, allowed_classifiers)
        cs.add_configuration_space("classifier", classifier_cs)

        return cs


# Custom function to restrict classifier types
def restrict_classifiers(
    classifiers: dict, allowed_classifiers: list
) -> ConfigurationSpace:
    restricted_classifiers = {
        key: value for key, value in classifiers.items() if key in allowed_classifiers
    }
    cs = ConfigurationSpace()
    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "__choice__", choices=list(restricted_classifiers.keys())
        )
    )
    for component in restricted_classifiers.values():
        cs.add_configuration_space(
            component.__name__, component.get_hyperparameter_search_space()
        )
    return cs


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

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

automl = CustomAutoSklearnClassifier(
    time_left_for_this_task=1800,
    per_run_time_limit=60,
    tmp_folder="/tmp/autosklearn_classification_example_tmp",
)

automl.fit(X_train, y_train)

print(automl.leaderboard())

pprint(automl.show_models(), indent=4)

predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))

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
