import subprocess

# List the packages you want to install
packages = ["numpy", "pandas", "liac-arff", "scikit-learn", "tensorflow", "matplotlib"]
# Run the pip install command and wait for it to complete
subprocess.run(["pip", "install"] + packages, check=True)
import tensorrt as rt
import tensorflow as tf
import numpy as np
import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, Nadam, Ftrl
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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


# Create model with each hyperparameter.
def create_model(optimizer, epochs, batch_size, learning_rate, dropout_rate):
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
    return model


# Define the parameter distribution for the search
param_dist = {
    "optimizer": [Adam],
    "learning_rate": [0.001, 0.0005, 0.0001],
    "dropout_rate": [0.3, 0.5, 0.7],
    "batch_size": [16, 32, 64],
    "epochs": [50, 100, 150],
}

# Create a KerasClassifier wrapper
model = KerasClassifier(build_fn=create_model, verbose=1)

use_gpu = bool(gpus)

# Set the number of jobs for parallel processing based on GPU availability
n_jobs = -1 if use_gpu else 1

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    cv=5,
    n_iter=10,
    n_jobs=n_jobs
)

# Fit the RandomizedSearchCV object to the data
random_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters found: ", random_search.best_params_)

# Create the callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=10)
model_checkpoint = ModelCheckpoint(
    "best_model.h5", monitor="val_accuracy", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5)

# Train the model with the best hyperparameters
best_model = random_search.best_estimator_

# Set the hyperparameters of best_model to the best parameters found by randomized search
best_model.set_params(**random_search.best_params_)

# Fit the model with the best hyperparameters
history = best_model.fit(
    X_train,
    y_train,
    epochs=random_search.best_params_["epochs"],
    batch_size=random_search.best_params_["batch_size"],
    validation_data=(X_test, y_test),
    callbacks=[early_stop, model_checkpoint, reduce_lr]
)

# Evaluate the model
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Make prediction
predictions = best_model.predict(X_test)

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

# Extract the training and validation loss values from the history object
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Create a dataframe to store the grid search results
results_df = pd.DataFrame(random_search.cv_results_)

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
results_df["test_accuracy"] = random_search.cv_results_["mean_test_score"]

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
