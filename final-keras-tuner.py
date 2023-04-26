import os
import tempfile
import arff
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras_tuner
from keras import backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

# Load the dataset
with open("phpgNaXZe.arff", "r") as f:
    dataset = arff.load(f)
data = np.array(dataset["data"])

raw_df = pd.DataFrame(data, columns=[attr[0] for attr in dataset["attributes"]])

cleaned_df = raw_df.copy()

# Replace all occurrences of 2 with 1, 1 with 2 in the "chd" column to binary fit the original label
cleaned_df["chd"] = cleaned_df["chd"].replace(["2", "1"], ["1", "0"])
# Replace all occurrences of 2 with 0 in the "famhist" column
cleaned_df["famhist"] = cleaned_df["famhist"].replace("2", "0")

# Panda recognises all as objects (string?)
print(cleaned_df.dtypes)

# Apply numeric
cleaned_df = cleaned_df.apply(pd.to_numeric, errors="coerce")

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

neg, pos = np.bincount(cleaned_df["chd"])
total = neg + pos
print(
    "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
        total, pos, 100 * pos / total
    )
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

# Makes all columns on a standard scale, by removing the mean and applying std of 0.

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

print("Training labels shape:", train_labels.shape)
print("Validation labels shape:", val_labels.shape)
print("Test labels shape:", test_labels.shape)

print("Training features shape:", train_features.shape)
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
def make_model(hp, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential()
    model.add(layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice(
                    "activation", ["relu", "tanh", "elu", "LeakyReLU"]
                ),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(1, activation="sigmoid", bias_initializer=output_bias))
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model


EPOCHS = 300
BATCH_SIZE = 64

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_prc", verbose=1, patience=10, mode="max", restore_best_weights=True
)

initial_bias = np.log([pos / neg])

make_model(keras_tuner.HyperParameters(), output_bias=initial_bias)

tuner = keras_tuner.RandomSearch(
    hypermodel=make_model,
    objective="val_accuracy",
    max_trials=100,
    executions_per_trial=3,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search_space_summary()
input()

tuner.search(
    train_features,
    train_labels,
    epochs=3,
    validation_data=(val_features, val_labels),
)

input()

model = make_model(output_bias=initial_bias)
model.summary()
model.predict(train_features[:10])
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Initial Bias", initial_bias)
print("Loss: {:0.4f}".format(results[0]))

initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")
model.save_weights(initial_weights)

model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels),
)

mpl.rcParams["figure.figsize"] = (12, 10)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_metrics(history):
    metrics = ["loss", "prc", "precision", "recall"]
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label="Train")
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=colors[0],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


plot_metrics(baseline_history)
plt.show()

train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix @{:.2f}".format(p))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    print("Correctly Predict no chd (True Negatives): ", cm[0][0])
    print("Falsely Predict chd (False Positives): ", cm[0][1])
    print("Falsely Predict no chd (False Negatives): ", cm[1][0])
    print("Correctly Predict chd (True Positives): ", cm[1][1])


baseline_results = model.evaluate(
    test_features, test_labels, batch_size=BATCH_SIZE, verbose=0
)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ": ", value)
print()

# Plot Confusion Matrix
plot_cm(test_labels, test_predictions_baseline)
plt.show()

# Calculate the class weights based on the imbalance ratio

weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))
weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels),
    # Class weights
    class_weight=class_weight,
)

plot_metrics(weighted_history)
plt.show()

train_predictions_weighted = weighted_model.predict(
    train_features, batch_size=BATCH_SIZE
)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)

weighted_results = weighted_model.evaluate(
    test_features, test_labels, batch_size=BATCH_SIZE, verbose=0
)
for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ": ", value)
print()

# Plot weighted confusion matrix.
plot_cm(test_labels, test_predictions_weighted)
plt.show()
