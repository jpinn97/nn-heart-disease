import subprocess

# List the packages you want to install
packages = ["numpy", "pandas", "liac-arff", "scikit-learn", "tensorflow", "matplotlib"]
# Run the pip install command and wait for it to complete
subprocess.run(["pip", "install"] + packages, check=True)

import tensorflow as tf
import numpy as np
import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, Nadam, Ftrl
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
with open("phpgNaXZe.arff", "r") as f:
    dataset = arff.load(f)
data = np.array(dataset["data"])
df = pd.DataFrame(data, columns=[attr[0] for attr in dataset["attributes"]])

# Preprocessing
X = df.iloc[:, :-1].values
y = (
    df.iloc[:, -1].values.astype(int) - 1
)  # Convert labels to integers and subtract 1 to have labels in {0, 1}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def enable_dynamic_memory_allocation():
    # Set the visible GPU devices
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                # Enable dynamic memory allocation
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Dynamic memory allocation enabled for", len(gpus), "GPU(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU devices found")


# Call the function to enable dynamic memory allocation for TensorFlow on the GPU
enable_dynamic_memory_allocation()

# Create model with each hyperparameter.
def create_model(optimizer, epochs, batch_size, learning_rate, dropout_rate):
    model = tf.keras.Sequential()
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    # Compile Model
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


# Create a KerasClassifier wrapper
model = KerasClassifier(build_fn=create_model, verbose=1)

# Define the parameter grid for the search
param_grid = {
    "optimizer": [Adam],
    "learning_rate": [0.001, 0.0005, 0.0001],
    "dropout_rate": [0.3, 0.5, 0.7],
    "batch_size": [16, 32, 64],
    "epochs": [50, 100, 150],
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters found: ", grid_result.best_params_)

# Train the model with the best hyperparameters
best_model = grid_result.best_estimator_

early_stop = EarlyStopping(monitor="val_loss", patience=10)

# Train the best model
history = best_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
)

# Evaluate the model

_, accuracy = best_model.model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f" % (accuracy * 100))

# Evaluate against test data
test_loss, test_accuracy = best_model.model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Make prediction
predictions = best_model.model.predict(X_test)

# Convert predictions to bianry
binary_predictions = (predictions > 0.5).astype(int)

# Analyze performance
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, binary_predictions))

# Print confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, binary_predictions).ravel()
print("Confusion Matrix:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Extract the training and validation loss values from the history object
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Create a dataframe to store the grid search results
results_df = pd.DataFrame(grid_result.cv_results_)

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
results_df["test_accuracy"] = grid_result.cv_results_["mean_test_score"]

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
