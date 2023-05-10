import subprocess
import os
import threading
import tensorflow as tf
import numpy as np
import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)
from imblearn.over_sampling import ADASYN

# List the packages you want to install
packages = [
    "numpy",
    "pandas",
    "liac-arff",
    "scikit-learn",
    "tensorflow",
    "matplotlib",
    "imblearn",
]
# Run the pip install command and wait for it to complete
# subprocess.run(["pip", "install"] + packages, check=True)

# Load the dataset
with open("phpgNaXZe.arff", "r") as f:
    dataset = arff.load(f)
data = np.array(dataset["data"])
df = pd.DataFrame(data, columns=[attr[0] for attr in dataset["attributes"]])

# Preprocessing
categorical_cols = ["famhist", "chd"]
numerical_cols = [
    "sbp",
    "tobacco",
    "ldl",
    "adiposity",
    "type",
    "obesity",
    "alcohol",
    "age",
]

X_categorical = pd.get_dummies(df[categorical_cols])
X_numerical = df[numerical_cols]
X = pd.concat([X_categorical, X_numerical], axis=1).values

y = df["chd"].values.astype(int) - 1  # Convert class labels to 0/1
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Perform oversampling on the training set to balance the class distribution
oversampler = ADASYN()
X_train, y_train = oversampler.fit_resample(X_train, y_train)

gpus = tf.config.list_physical_devices("GPU")


def enable_dynamic_memory_allocation():
    # Set the GPU as the default device
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.set_visible_devices(gpus[0], "GPU")
            # Set the memory growth option to True
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("Using GPU:", gpus[0].name)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPUs found.")


# Call the function to enable dynamic memory allocation for TensorFlow on the GPU
# enable_dynamic_memory_allocation()

# Create the callbacks
early_stop = EarlyStopping(monitor="loss", patience=10)
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5)


# Create model with fixed hyperparameters
def create_model():
    model = Sequential()
    model.add(Dense(9, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    # Compile Model
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"],
    )
    # Add Callbacks
    model.callbacks = [early_stop, reduce_lr]

    return model


# Create a KerasClassifier wrapper
model = KerasClassifier(build_fn=create_model, verbose=1)

# Fit the model with the fixed hyperparameters
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
)

# Evaluate the model
test_loss, test_accuracy = model.model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Make prediction
predictions = model.predict(X_test)

# Get the predicted probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Convert predictions to binary
binary_predictions = (predictions > 0.5).astype(int)

# Analyze performance
print("Classification Report:")
print(classification_report(y_test, binary_predictions))

# Print confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, binary_predictions).ravel()
print("Confusion Matrix:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_prob)
print("AUC Score:", auc_score)

# Extract the training and validation loss values from the history object
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Plot the training and validation loss values over epochs
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
