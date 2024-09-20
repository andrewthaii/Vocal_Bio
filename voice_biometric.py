import speech_recognition as sr
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import os



# Function to extract MFCC features from the audio file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=None)  # Use the sample rate in the file itself
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting features from {file_name}: {e}")
        return None

# Function to train the model using your voice samples and print features for all the files
def train_model():
    voice_samples_dir = "voice_samples"
    
    X = []  # To store feature vectors
    y = []  # To store labels (1 for your voice, 0 for others)
    files = []  # To store the filenames for debugging

    # Go through each file in the 'voice_samples' directory
    for file in os.listdir(voice_samples_dir):
        if file.endswith(".wav"):
            print(f"Processing file: {file}")  # Notify that the file is being processed
            
            # Label the file based on whether it's your voice or someone else's
            label = 1 if "your_voice" in file else 0  # 1 for your voice, 0 for others
            
            # Extract features from the current audio file
            file_path = os.path.join(voice_samples_dir, file)
            features = extract_features(file_path)
            
            if features is not None:  # Make sure feature extraction succeeded
                X.append(features)
                y.append(label)
                files.append(file)
                print(f"Features for {file} - Label: {label}")
                print(f"MFCC Features: {features}\n")  # Print the extracted features (MFCCs)
            else:
                print(f"Failed to extract features from {file}")
    
    # Check if we have 6 data points (since you have 6 training files)
    if len(X) != 6:
        print(f"Error: Expected 6 training samples but got {len(X)}. Please check your 'voice_samples' folder.")
        return None, None, None

    # Train a K-Nearest Neighbors classifier using all data (no train/test split)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)  # Use all data for training
    
    # Print a summary of the extracted features, labels, and files
    print("\n### Summary of Features and Labels ###")
    for i, features in enumerate(X):
        print(f"File: {files[i]}, Label: {y[i]}, Features: {features}")
    
    return model, X, y  # Return all data as X_train, y_train for testing later


# Function to recognize your voice in pre-recorded test files and print Euclidean distances, along with X_train and y_train
def recognize_voice(model, X_train, y_train):
    # List of test files to be recognized
    test_files = ["test_voice_1.wav", "test_voice_2.wav", "test_voice_3.wav", "test_voice_4.wav", "test_voice_5.wav", "test_voice_6.wav"]

    for test_file in test_files:
        print(f"\nRecognizing voice in file: {test_file}")

        # Extract features from the test file
        features = extract_features(test_file)

        if features is not None:  # Ensure feature extraction succeeded
            print(f"Features for {test_file}: {features}")

            # Calculate Euclidean distance to each of the training samples and print X_train and y_train
            for i, (train_features, label) in enumerate(zip(X_train, y_train)):
                label_type = "your_voice" if label == 1 else "other_voice"
                distance = euclidean(features, train_features)
                
                # Print the training sample's feature vector and its corresponding label
                print(f"\nX_train[{i}]: {train_features}")
                print(f"y_train[{i}]: {label} ({label_type})")
                
                # Print the Euclidean distance between the test file and the current training sample
                print(f"Distance from {label_type}_{i+1}: {distance:.4f}")

            # Predict the label (0 or 1)
            result = model.predict([features])

            # Check the result and print the outcome
            if result == 1:
                print(f"{test_file}: This is your voice!")
            else:
                print(f"{test_file}: This is not your voice.")
        else:
            print(f"Failed to extract features from {test_file}")

if __name__ == "__main__":
    print("### Voice Biometric Project ###")

    # Train the model and get training data
    model, X_train, y_train = train_model()

    # If the model is successfully trained, recognize test voices
    if model is not None:
        # Pass both the model and training data (X_train, y_train) to the recognize_voice function
        recognize_voice(model, X_train, y_train)
