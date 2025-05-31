import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 🔍 Define dataset path
DATA_PATH = f"D:\\ml assignment\\pro4\\Data\\genres_original"

# 🎵 Function to extract MFCC features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)  # Load 30s of audio
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)  # Take mean across time
        return mfccs_mean
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# 📂 Function to load dataset
def load_dataset(data_path):
    dataset = []
    genres = os.listdir(data_path)

    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        if not os.path.isdir(genre_path):
            continue  # Skip if not a directory

        print(f"📂 Loading genre: {genre}")

        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)

            # Extract features
            features = extract_features(file_path)
            if features is not None:
                dataset.append([file] + list(features) + [genre])

    # Define column names
    columns = ["Filename"] + [f"MFCC_{i}" for i in range(13)] + ["Genre"]
    df = pd.DataFrame(dataset, columns=columns)

    if df.empty:
        print("⚠️ No features extracted! Check file paths and formats.")
    else:
        print(f"✅ Loaded {len(df)} audio samples.")

    return df

# 📊 Function to apply PCA
def apply_pca(df, n_components=5):
    X = df.iloc[:, 1:-1].values  # Extract MFCC features
    y = df.iloc[:, -1].values  # Genre labels

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"🔄 PCA explained variance: {pca.explained_variance_ratio_}")
    return X_pca, y, pca, scaler

# 🎯 Function to train and evaluate SVM
def train_and_evaluate_svm(X_pca, y):
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Train SVM
    svm_model = SVC(kernel='rbf', C=10, gamma='scale')
    svm_model.fit(X_train, y_train)

    # Predictions
    y_pred = svm_model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ SVM Accuracy: {accuracy:.2f}")

    # Classification Report
    print("\n📜 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return svm_model

# 🚀 Run the full pipeline
if __name__ == "__main__":
    print("🔄 Extracting features and loading dataset...")
    df = load_dataset(DATA_PATH)

    if not df.empty:
        print("📉 Applying PCA for dimensionality reduction...")
        X_pca, y, pca, scaler = apply_pca(df)

        print("⚡ Training SVM classifier...")
        model = train_and_evaluate_svm(X_pca, y)

        print("✅ Model training complete!")
    else:
        print("❌ Dataset could not be loaded. Check file paths.")
