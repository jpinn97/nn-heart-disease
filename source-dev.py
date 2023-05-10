import subprocess

# List the packages you want to install
packages = ["numpy", "pandas", "liac-arff", "scikit-learn", "tensorflow", "matplotlib"]
# Run the pip install command and wait for it to complete
subprocess.run(["pip", "install"] + packages, check=True)

import numpy as np
import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from scikeras.wrappers import KerasClassifier
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


def create_model(learning_rate=0.001, dropout_rate=0.5):
    model = Sequential()
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
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


clf = KerasClassifier(
    model=create_model,
    loss="binary_crossentropy",
    optimizer="adam",
    optimizer__lr=0.1,
    model__hidden_layer_sizes=(100,),
    model__dropout=0.5,
    verbose=False,
)

params = {
    "optimizer__lr": [0.05, 0.1],
    "model__hidden_layer_sizes": [
        (100,),
        (
            50,
            50,
        ),
    ],
    "model__dropout": [0, 0.5],
}

gs = GridSearchCV(clf, params, scoring="accuracy", n_jobs=-1, verbose=True)

gs.fit(X_train, y_train)

print(gs.best_score_, gs.best_params_)

# Print the best hyperparameters
print("Best parameters found: ", gs.best_params_)

# Train the model with the best hyperparameters
best_model = gs.best_estimator_

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

# Plot the training and validation loss values over epochs
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
