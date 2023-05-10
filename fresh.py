import arff
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the dataset
with open("phpgNaXZe.arff", "r") as f:
    dataset = arff.load(f)
data = np.array(dataset["data"])

# Split the dataset into features (X) and target variable (y)
X = data[:, :-1]
y = data[:, -1]
y = y.astype(int) - 1

# Encode the famhist attribute using one-hot encoding
famhist_col = X[:, 4].reshape(-1, 1)  # reshape to 2D array for one-hot encoding
encoder = OneHotEncoder(sparse=False)
famhist_encoded = encoder.fit_transform(famhist_col)

# Replace the original famhist attribute with the encoded version
X_encoded = np.hstack((X[:, :4], famhist_encoded, X[:, 5:])).astype(float)
y = y.astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Identify the indices of the continuous numerical attributes
columns_to_normalize = [0, 1, 2, 3, 5, 6, 7, 8]

# Create a mask to identify the continuous numerical attributes
mask = np.zeros(X_encoded.shape[1], dtype=bool)
mask[columns_to_normalize] = True

# Create the StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the selected columns in the training data
scaler.fit(X_train[:, mask])

# Transform the selected columns in both the training and testing data using the fitted scaler
X_train[:, mask] = scaler.transform(X_train[:, mask])
X_test[:, mask] = scaler.transform(X_test[:, mask])

# Define the oversampling and undersampling methods
oversample = SMOTE()
undersample = RandomUnderSampler()

# Create a pipeline to apply the resampling techniques
pipeline = Pipeline(steps=[("o", oversample), ("u", undersample)])

# Apply the pipeline to the training data
X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

# Split the resampled training data into training and validation sets
(
    X_train_resampled,
    X_val_resampled,
    y_train_resampled,
    y_val_resampled,
) = train_test_split(
    X_train_resampled, y_train_resampled, test_size=0.2, random_state=42
)

# Create the StandardScaler instance for the resampled data
scaler_resampled = StandardScaler()

# Fit the scaler on the resampled training data
scaler_resampled.fit(X_train_resampled[:, mask])

# Transform the training, validation, and testing data using the fitted scaler for the resampled data
X_train_resampled[:, mask] = scaler_resampled.transform(X_train_resampled[:, mask])
X_val_resampled[:, mask] = scaler_resampled.transform(X_val_resampled[:, mask])
X_test[:, mask] = scaler_resampled.transform(X_test[:, mask])

METRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
]


# Define a function that creates and compiles the model
def create_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential(
        [
            keras.layers.Dense(
                16, activation="relu", input_shape=(X_train_resampled.shape[1],)
            ),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias),
        ]
    )
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model


keras_classifier = KerasClassifier(
    build_fn=create_model, batch_size=256, epochs=100, verbose=1
)

cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
cross_val_scores = cross_val_score(
    keras_classifier, X_train_resampled, y_train_resampled, cv=cv
)

history = keras_classifier.fit(
    X_train_resampled,
    y_train_resampled,
    validation_data=(X_val_resampled, y_val_resampled),
)

print(
    "Cross-Validation Accuracy of Training: %0.2f (+/- %0.2f)"
    % (cross_val_scores.mean(), cross_val_scores.std() * 2)
)

test_metrics = keras_classifier.model.evaluate(X_test, y_test)

test_loss, test_acc = test_metrics[0], test_metrics[1]
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

y_pred = keras_classifier.predict(X_test)
y_pred_classes = np.round(y_pred)

conf_matrix = confusion_matrix(y_test, y_pred_classes)
conf_matrix = np.flip(conf_matrix, axis=0)

print(conf_matrix)

report = classification_report(y_test, y_pred_classes, output_dict=True)
print(classification_report(y_test, y_pred_classes))

fig, axs = plt.subplots(ncols=3, figsize=(18, 6))

sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", ax=axs[0], cbar=False)
axs[0].set_title("Confusion Matrix")

sns.heatmap(
    pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Blues", ax=axs[1], cbar=False
)
axs[1].set_title("Classification Report")

# Extract the training and validation loss values from the history object
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Plot the training and validation loss values over epochs
epochs = range(1, len(train_loss) + 1)
axs[2].plot(epochs, train_loss, "bo", label="Training Loss")
axs[2].plot(epochs, val_loss, "b", label="Validation Loss")
axs[2].set_title("Training and Validation Loss")
axs[2].set_xlabel("Epochs")
axs[2].set_ylabel("Loss")
axs[2].legend()

plt.tight_layout()
plt.show()
