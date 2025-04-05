import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, GaussianNoise

# Create a folder to save results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

files = ["emg_healthy", "emg_myopathy", "emg_neuropathy"]

def load_emg_signal(filename):
    """Load EMG signal using wfdb"""
    record = wfdb.rdrecord(filename)
    signal = record.p_signal[:, 0]
    return signal

def preprocess_data(files, sequence_length=300):
    """Load, normalize, and create sequences for CNN-LSTM"""
    X, y = [], []
    scaler = StandardScaler()
    
    for i, file in enumerate(files):
        signal = load_emg_signal(file)
        signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        for j in range(0, len(signal) - sequence_length, sequence_length):
            X.append(signal[j:j+sequence_length])
            y.append(i)
    
    return np.array(X), np.array(y)

X, y = preprocess_data(files)
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Sequential([
    GaussianNoise(0.01, input_shape=(X_train.shape[1], 1)),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save accuracy to a file
accuracy_file = os.path.join(results_dir, "accuracy.txt")
with open(accuracy_file, "w") as f:
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("CNN-LSTM Classification Performance")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(results_dir, "training_performance.png"))  # Save plot
plt.show()

# Save the model
model.save(os.path.join(results_dir, "emg_model.h5"))

