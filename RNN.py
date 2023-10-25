import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# Load the data from the CSV file
data = pd.read_csv("extracted_features.csv")

# Extract features and labels
X = data.iloc[:, :-1].values  # Features (all columns except the last one)
y = data.iloc[:, -1].values   # Class labels (the last column)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode class labels if they are categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Create sequences from the data
sequence_length = 10  # You can choose an appropriate sequence length
num_features = X.shape[1]

# Create sequences and labels
sequences = []
labels = []
for i in range(len(X) - sequence_length + 1):
    sequences.append(X[i:i + sequence_length])
    labels.append(y[i + sequence_length - 1])

X_sequences = np.array(sequences)
y_labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_labels, test_size=0.2, random_state=42)

# Define the RNN model
model = keras.Sequential()
model.add(SimpleRNN(units=64, input_shape=(sequence_length, num_features), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=np.max(y_labels) + 1, activation='softmax'))  # Assumes y_labels contains class labels

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=2000, batch_size=32)

# Evaluate the model on the existing testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Calculate predictions for the test data
y_pred = model.predict(X_test)

# Convert predictions from one-hot encoded format to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert true class labels to a flat array
y_true = y_test

# Create a confusion matrix
confusion = confusion_matrix(y_true, y_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

newTestingData = pd.read_csv("Testing_Data.csv")

# Normalize features
X_test = newTestingData.iloc[:, :-1].values
X_test = scaler.transform(X_test)

# Encode class labels if they are categorical
y_test = newTestingData.iloc[:, -1].values
y_test = label_encoder.transform(y_test)

sequences_test = []
labels_test = []
for i in range(len(X_test) - sequence_length + 1):
    sequences_test.append(X_test[i:i + sequence_length])
    labels_test.append(y_test[i + sequence_length - 1])

X_sequences_test = np.array(sequences_test)
y_labels_test = np.array(labels_test)

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(X_sequences_test, y_labels_test)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Calculate predictions for the test data
y_pred_test = model.predict(X_sequences_test)

# Convert predictions from one-hot encoded format to class labels
y_pred_labels_test = np.argmax(y_pred_test, axis=1)

# Convert true class labels to a flat array
y_true_test = y_labels_test

# Create a confusion matrix
confusion_test = confusion_matrix(y_true_test, y_pred_labels_test)

# Display the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_test, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Testing Data')
plt.show()


# Display the confusion matrix using seaborn
