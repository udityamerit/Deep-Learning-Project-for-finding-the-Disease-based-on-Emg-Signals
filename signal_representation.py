import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt

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

# Function to plot the EMG signal
def plot_emg_signal(file_path):
    """Loads and plots the EMG signal from a .dat or .hea file"""
    signal = load_emg_signal(file_path)
    
    if signal is None:
        print("Error: Unable to load the signal. Please check the file.")
        return
    
    # Generate time axis
    time = np.arange(len(signal))  # Assuming a uniform sampling rate
    
    # Plot the EMG signal
    plt.figure(figsize=(12, 5))
    plt.plot(time, signal, label="EMG Signal", color="blue")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title(f"EMG Signal from {os.path.basename(file_path)}")
    plt.legend()
    plt.grid()
    plt.show()

# Main function to take user input and plot the EMG signal
if __name__ == "__main__":
    while True:
        print("\nOptions:")
        print("1. Enter a file path to plot EMG signal")
        print("2. Exit")
        choice = input("Enter your choice (1/2): ").strip()

        if choice == "1":
            file_path = input("Enter the full file path (.dat or .hea): ").strip()

            if not os.path.exists(file_path):
                print("Error: The file path does not exist. Please check the file path.")
            else:
                plot_emg_signal(file_path)

        elif choice == "2":
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please enter 1 or 2.")
            
