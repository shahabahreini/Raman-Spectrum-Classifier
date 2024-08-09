# 1. Import Libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scikitplot as skplt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint


matplotlib.rcParams['savefig.dpi'] = 1024
matplotlib.rcParams["figure.dpi"] = 150

# 2. Load CSV files
file1 = "Rab2.csv"
file2 = "Wt3.csv"

df1 = pd.read_csv(file1,  header=None).T
df2 = pd.read_csv(file2,  header=None).T

# Create labels for the samples (e.g., 0 for file1 and 1 for file2)
labels1 = np.zeros(df1.shape[0])
labels2 = np.ones(df2.shape[0])

X = pd.concat([df1, df2])
y = np.concatenate([labels1, labels2])

# 3. Preprocess the Data
# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build the Model
model = Sequential()
model.add(Dense(80, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer="adam", metrics=["accuracy"])
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Create a checkpoint callback
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# 5. Train the Model
# Pass the checkpoint callback to the fit method
model.fit(X_train, y_train, epochs=300, batch_size=4, validation_data=(X_val, y_val), callbacks=[checkpoint])


# Load the best model
model.load_weights('best_model.h5')

# 6. Evaluate the Model
y_pred_proba = model.predict(X_val)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

flist = [file1, file2]
labels = [fname.replace(".csv", "").upper() for fname in flist]
conf_matrix = confusion_matrix(y_val, y_pred)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels).plot()
plt.show()
skplt.metrics.plot_confusion_matrix(y_val, y_pred, normalize=False, title='Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.clf()

accuracy = accuracy_score(y_val, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)

# After training and evaluating the model, we can compute permutation importance

# Store the original accuracy
original_accuracy = accuracy

# Initialize an array to store the importances
importances = []

# Loop through each feature
for i in range(X_train.shape[1]):
    # Create a copy of the validation data
    X_val_permuted = X_val.copy()

    # Permute the values of the i-th feature
    np.random.shuffle(X_val_permuted[:, i])

    # Predict using the permuted data
    y_pred_proba_permuted = model.predict(X_val_permuted)
    y_pred_permuted = (y_pred_proba_permuted > 0.5).astype(int).flatten()

    # Compute the accuracy using the permuted data
    permuted_accuracy = accuracy_score(y_val, y_pred_permuted)

    # Compute the importance as the difference between the original and permuted accuracy
    importance = original_accuracy - permuted_accuracy
    importances.append(importance)

# Print the importances
for i, importance in enumerate(importances):
    print(f"Feature {i}: {importance}")

# You can also visualize the importances
feature_name = pd.read_csv("Spectrum_650.csv", header=None)
feature_name = feature_name.iloc[:, 0].tolist()

plt.bar(feature_name, importances)
# plt.bar(range(len(importances)), importances)
plt.xlabel("Spectrum")
plt.ylabel("Importance")
plt.savefig("importance.png")
plt.show()
