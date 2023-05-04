import json
import os
import tempfile
import arff
import numpy as np
import pandas as pd
import tensorboard
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from imblearn.over_sampling import SMOTE
import keras.layers
from keras_tuner import HyperParameters, Objective
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import seaborn as sns
import keras_tuner

# Check if GPU is available
print(
    "GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE"
)
# Check if CPU is available
print(
    "CPU is", "available" if tf.config.list_physical_devices("CPU") else "NOT AVAILABLE"
)

num_physical_cores = os.cpu_count()
num_sockets = 1

# Set TensorFlow configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.allow_soft_placement = True
config.intra_op_parallelism_threads = num_physical_cores
config.inter_op_parallelism_threads = num_sockets
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

if matplotlib.get_backend() == "agg":
    matplotlib.use("TkAgg")

# Load the dataset
with open("phpgNaXZe.arff", "r") as f:
    dataset = arff.load(f)
data = np.array(dataset["data"])

raw_df = pd.DataFrame(data, columns=[attr[0] for attr in dataset["attributes"]])

cleaned_df = raw_df.copy()

# Replace all occurrences of 2 with 1, 1 with 0 in the "chd" column to binary fit the original label
cleaned_df["chd"] = cleaned_df["chd"].replace(["2", "1"], ["1", "0"])
# Replace all occurrences of 2 with 0 in the "famhist" column
cleaned_df["famhist"] = cleaned_df["famhist"].replace("2", "0")

# Panda recognises all as objects (string?)
print(cleaned_df.dtypes)
# Apply numeric
cleaned_df = cleaned_df.apply(pd.to_numeric, errors="coerce")

print(cleaned_df.isna().sum())

# Check again
print(cleaned_df.dtypes)

# Print descriptive statistics
print(
    cleaned_df[
        [
            "sbp",
            "tobacco",
            "ldl",
            "adiposity",
            "famhist",
            "type",
            "obesity",
            "alcohol",
            "age",
            "chd",
        ]
    ].describe()
)

# Create a list of columns to perform outlier detection on. (Not Target chd, or famihist)
outlier_columns = ["sbp", "tobacco", "ldl", "adiposity", "type", "obesity", "alcohol"]

# Calculate the Z-scores for each value in the outlier_columns
z_scores = np.abs(
    (cleaned_df[outlier_columns] - cleaned_df[outlier_columns].mean())
    / cleaned_df[outlier_columns].std()
)

# Set the threshold for outliers (e.g., 3)
threshold = 3

# Find the median values for each column in outlier_columns
medians = cleaned_df[outlier_columns].median()

# Replace the outliers with the corresponding median value
cleaned_df_no_outliers = cleaned_df.copy()
cleaned_df_no_outliers[outlier_columns] = cleaned_df[outlier_columns].mask(
    z_scores > threshold, medians, axis=1
)

# Create a 1x2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plot for raw data
cleaned_df.boxplot(
    column=[
        "sbp",
        "tobacco",
        "ldl",
        "adiposity",
        "famhist",
        "type",
        "obesity",
        "alcohol",
        "age",
        "chd",
    ],
    ax=axes[0],
)
axes[0].set_title("Box plot of numeric columns - raw")
axes[0].set_xlabel("Features")
axes[0].set_ylabel("Value")

# Box plot for cleaned data
cleaned_df_no_outliers.boxplot(
    column=[
        "sbp",
        "tobacco",
        "ldl",
        "adiposity",
        "famhist",
        "type",
        "obesity",
        "alcohol",
        "age",
        "chd",
    ],
    ax=axes[1],
)
axes[1].set_title("Box plot of numeric columns - replaced outliers with median")
axes[1].set_xlabel("Features")
axes[1].set_ylabel("Value")

# Adjust the space between the subplots
plt.subplots_adjust(wspace=0.3)

# Show the plot
plt.show()

# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(cleaned_df_no_outliers, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop("chd"))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop("chd"))
test_labels = np.array(test_df.pop("chd"))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
train_features_resampled, train_labels_resampled = smote.fit_resample(
    train_features, train_labels
)

# Makes all columns on a standard scale, by removing the mean and applying std of 0.

scaler = StandardScaler()
train_features_resampled = scaler.fit_transform(train_features_resampled)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

print("Training labels shape:", train_labels_resampled.shape)
print("Validation labels shape:", val_labels.shape)
print("Test labels shape:", test_labels.shape)

print("Training features shape:", train_features_resampled.shape)
print("Validation features shape:", val_features.shape)
print("Test features shape:", test_features.shape)

# Plot histograms of all features except 'chd' and 'famhist', 'adiposity', 'age'
features = ["sbp", "tobacco", "ldl", "type", "obesity", "alcohol"]

plt.figure(figsize=(16, 8))
plt.suptitle("Distribution of Numeric Features")

for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data=cleaned_df, x=feature, kde=True, stat="density")
    sns.histplot(data=cleaned_df_no_outliers, x=feature, kde=True, stat="density")
    plt.title(feature)
    plt.legend(["Original", "No Outliers"])
    plt.xlabel("")

plt.tight_layout()
plt.show()

# Plot 2: Correlation heatmaps
plt.figure(figsize=(14, 6))
plt.suptitle("Correlation Heatmap")

plt.subplot(1, 2, 1)
corr_matrix_original = cleaned_df.corr()
sns.heatmap(corr_matrix_original, annot=True, cmap="coolwarm")
plt.title("Original")

plt.subplot(1, 2, 2)
corr_matrix_no_outliers = cleaned_df_no_outliers.corr()
sns.heatmap(corr_matrix_no_outliers, annot=True, cmap="coolwarm")
plt.title("No Outliers")

plt.tight_layout()
plt.show()

# Calculate class ratio and weights for original training data
train_targets = cleaned_df["chd"].values
counts = np.bincount(train_targets)
print("Original Training Data:")
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(train_targets)
    )
)
class_weights = compute_class_weight("balanced", classes=[0, 1], y=train_targets)
class_weights = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weights)

# Calculate class ratio and weights for resampled training data
train_targets_resampled = train_labels_resampled
counts_resampled = np.bincount(train_targets_resampled)
print("Resampled Training Data:")
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts_resampled[1],
        100 * float(counts_resampled[1]) / len(train_targets_resampled),
    )
)
class_weights_resampled = compute_class_weight(
    "balanced", classes=[0, 1], y=train_targets_resampled
)
class_weights_resampled = {
    i: weight for i, weight in enumerate(class_weights_resampled)
}
print("Class weights:", class_weights_resampled)

# Keras Evaluation metrics

METRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    # Possibly remove accuracy
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
]


# Construct model
def build_model(hp, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential()
    model.add(
        layers.Dense(
            units=hp.Int(f"units_{0}", min_value=8, max_value=512, step=32),
            activation=hp.Choice("activation", ["relu", "tanh", "elu", "LeakyReLU"]),
            input_shape=(train_features_resampled.shape[-1],),
        ),
    )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=8, max_value=512, step=16),
                activation=hp.Choice(
                    "activation", ["relu", "tanh", "elu", "LeakyReLU"]
                ),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(1, activation="sigmoid", bias_initializer=output_bias))

    learning_rate = hp.Float("lr", min_value=5e-5, max_value=1e-2, sampling="log")

    # Add optimizer choice to hyperparameters
    optimizer_choice = hp.Choice("optimizer", ["adam", "sgd", "rmsprop"])

    # Set up optimizer based on choice
    if optimizer_choice == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_choice == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS,
    )

    return model


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)

# Count labels
positive_count = np.sum(train_labels_resampled)
total_count = len(train_labels_resampled)
negative_count = total_count - positive_count

# Calculate the ratio of the positive samples to the negative samples
ratio = positive_count / negative_count

# Calculate the initial bias
initial_bias = np.log(ratio)

print("Initial bias:", initial_bias)

build_model(HyperParameters(), output_bias=initial_bias)

# Specify the objective
objectives = [
    Objective("val_loss", direction="min"),
    Objective("val_accuracy", direction="max"),
    Objective("val_auc", direction="max"),
]

tuner = keras_tuner.BayesianOptimization(
    hypermodel=lambda hp: build_model(hp, output_bias=initial_bias),
    objective=objectives,
    max_trials=1,
    executions_per_trial=2,
    tune_new_entries=True,
    allow_new_entries=True,
    seed=42,
    overwrite=False,
    directory="my_dir",
    project_name="bayesian_optimization",
)

tuner.search_space_summary()

batch_size = 64

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
'''
tuner.search(
    train_features_resampled,
    train_labels_resampled,
    epochs=300,
    class_weight=class_weights,
    batch_size=batch_size,
    callbacks=[early_stopping, tensorboard_callback],
    validation_data=(val_features, val_labels),
)
'''
tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]

evaluation_metrics = best_model.evaluate(test_features, test_labels)
test_loss = evaluation_metrics[0]
test_accuracy = evaluation_metrics[1]

# Compute the predicted labels for the test set
predicted_labels = (best_model.predict(test_features) > 0.5).astype(int)

# Compute the test accuracy, ROC AUC, precision, recall, and F1 score based on the true and predicted labels
accuracy = accuracy_score(test_labels, predicted_labels)
roc_auc = roc_auc_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)

# Compute the precision-recall curve and ROC curve
precision_curve, recall_curve, _ = precision_recall_curve(test_labels, predicted_labels)
fpr, tpr, _ = roc_curve(test_labels, predicted_labels)

# Compute the area under the precision-recall curve and ROC curve
prc_auc = auc(recall_curve, precision_curve)
roc_auc = auc(fpr, tpr)

# Compute the confusion matrix based on the true and predicted labels
cm = confusion_matrix(test_labels, predicted_labels)

# Plot the confusion matrix using seaborn
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Model {i + 1} Confusion Matrix")
plt.show()

# Print the evaluation metrics for the model
print(f"\nModel {i + 1} evaluation:")
print(f"Test accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PRC AUC: {prc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1:.4f}")
