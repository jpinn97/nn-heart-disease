import subprocess

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
subprocess.run(["pip", "install"] + packages, check=True)
import os
import threading
import tensorflow as tf
import numpy as np
import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorboard import program
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, Nadam, Ftrl
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
from imblearn.over_sampling import RandomOverSampler

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

# Perform oversampling on the training set to balance the class distribution
oversampler = RandomOverSampler()
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
enable_dynamic_memory_allocation()

# Create the callbacks
early_stop = EarlyStopping(monitor="loss", patience=10)
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5)

# Start TensorBoard

log_dir = "./logs"

def start_tensorboard(log_dir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_dir])
    url = tb.launch()
    print(f"TensorBoard is running at {url}")


thread = threading.Thread(target=start_tensorboard, args=(log_dir,))
thread.start()
tensorboard = tensorboard = TensorBoard(log_dir=log_dir)


# Create model with each hyperparameter.
def create_model(optimizer, learning_rate, dropout_rate, callbacks):
    model = Sequential()
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    # Compile Model
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    # Add Callbacks
    model.callbacks = callbacks

    return model


# Define the parameter distribution for the search
param_dist = {
    "optimizer": [Adam, RMSprop, Adadelta, Adagrad, Nadam, Ftrl],
    "learning_rate": [0.001, 0.0005, 0.0001],
    "dropout_rate": [0.3, 0.5, 0.7],
    "batch_size": [16, 32, 64],
    "epochs": [50, 100, 150],
    "callbacks": [early_stop, reduce_lr, tensorboard],
}


# Create a KerasClassifier wrapper
model = KerasClassifier(build_fn=create_model, verbose=1)

use_gpu = bool(gpus)

# Set the number of jobs for parallel processing based on GPU availability
n_jobs = 1 if use_gpu else -1

# Create a RandomizedSearchCV object
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_dist,
    cv=5,
    n_jobs=n_jobs,
)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train, callbacks=[early_stop, reduce_lr, tensorboard])

# Print the best hyperparameters
print("Best parameters found: ", grid_search.best_params_)

# Train the model with the best hyperparameters
best_model = grid_search.best_estimator_

# Set the hyperparameters of best_model to the best parameters found by randomized search
best_model.set_params(**grid_search.best_params_)

from keras.models import save_model

# Save best_model weights.
save_model(best_model.model, "best_model_save", save_format="tf")

# Fit the model with the best hyperparameters
history = best_model.fit(
    X_train,
    y_train,
    epochs=grid_search.best_params_["epochs"],
    batch_size=grid_search.best_params_["batch_size"],
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr, tensorboard],
)

# Evaluate the model
test_loss, test_accuracy = best_model.model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Make prediction
predictions = best_model.predict(X_test)

# Get the predicted probabilities
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

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

# Create a dataframe to store the grid search results
results_df = pd.DataFrame(grid_search.cv_results_)

# Add a column to store the optimizer name
results_df["optimizer_name"] = results_df["params"].apply(
    lambda x: x["optimizer"].__name__
)

# Add columns to store the hyperparameters
results_df["learning_rate"] = results_df["params"].apply(lambda x: x["learning_rate"])
results_df["dropout_rate"] = results_df["params"].apply(lambda x: x["dropout_rate"])
results_df["batch_size"] = results_df["params"].apply(lambda x: x["batch_size"])
results_df["epochs"] = results_df["params"].apply(lambda x: x["epochs"])

# Add a column to store the test accuracy
results_df["test_accuracy"] = grid_search.cv_results_["mean_test_score"]

# Save the results to a CSV file
results_df.to_csv("grid_search_results.csv", index=False)

# Plot the training and validation loss values over epochs
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
