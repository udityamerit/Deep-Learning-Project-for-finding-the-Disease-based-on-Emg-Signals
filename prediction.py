import numpy as np
import tensorflow as tf
import wfdb
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Load the trained model
model = tf.keras.models.load_model("results/emg_model.h5")

# Create a folder to save results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Class labels
class_names = ["Healthy", "Myopathy", "Neuropathy"]

# Function to extract record name from .dat or .hea file
def get_record_name(file_path):
    """Extracts record name without extension"""
    return os.path.splitext(file_path)[0] if file_path.endswith((".dat", ".hea")) else file_path

# Function to load EMG signal from a given file
def load_emg_signal(file_path):
    """Loads EMG signal using wfdb"""
    record_name = get_record_name(file_path)  # Get base record name
    try:
        record = wfdb.rdrecord(record_name)  # Read the WFDB record
        signal = record.p_signal[:, 0]  # Extract first channel
        return signal
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Function to preprocess the EMG signal
def preprocess_signal(signal, sequence_length=300):
    """Normalizes and reshapes the signal for model input"""
    scaler = StandardScaler()
    signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    # Creating sequences
    sequences = [signal[i:i+sequence_length] for i in range(0, len(signal) - sequence_length, sequence_length)]

    if len(sequences) == 0:
        print("Error: Signal is too short for the model's input length.")
        return None
    
    return np.array(sequences).reshape(len(sequences), sequence_length, 1)

# Function to predict the EMG class
def predict_emg_class():
    """User enters filename, loads, preprocesses, and predicts EMG class"""
    file_path = input("Enter the full file path (.dat or .hea): ").strip()  # Get full file path from the user
    if not os.path.exists(file_path):
        print("Error: The file path does not exist. Please check the file path.")
        return

    signal = load_emg_signal(file_path)
    
    if signal is None:
        print("Error: Unable to load the signal. Please check the file.")
        return
    
    processed_signal = preprocess_signal(signal)
    if processed_signal is None:
        return
    
    # Make predictions
    predictions = model.predict(processed_signal)
    predicted_class = np.argmax(np.mean(predictions, axis=0))  # Averaging predictions across sequences
    
    print(f"\nPredicted Class: {class_names[predicted_class]}\n")

# Function to load and preprocess multiple test files
def load_test_data(test_files):
    """Loads and preprocesses multiple EMG files for evaluation"""
    X, y = [], []
    
    for i, file in enumerate(test_files):
        signal = load_emg_signal(file)
        if signal is None:
            continue
        processed_signal = preprocess_signal(signal)
        if processed_signal is None:
            continue
        
        X.extend(processed_signal)  # Add sequences
        y.extend([i] * len(processed_signal))  # Add corresponding labels
    
    return np.array(X), np.array(y)

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    """Plots confusion matrix and saves it"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))  # Save confusion matrix
    plt.show()

# Main execution
if __name__ == "__main__":
    while True:
        print("\nOptions:")
        print("1. Predict a single EMG signal")
        print("2. Evaluate model with test data & show confusion matrix")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            predict_emg_class()
        elif choice == "2":
            test_files = ["emg_healthy.dat", "emg_myopathy.dat", "emg_neuropathy.dat"]
            X_test, y_test = load_test_data(test_files)

            if len(X_test) > 0:
                y_pred = np.argmax(model.predict(X_test), axis=1)
                plot_confusion_matrix(y_test, y_pred)
            else:
                print("No valid test data found.")
        elif choice == "3":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")