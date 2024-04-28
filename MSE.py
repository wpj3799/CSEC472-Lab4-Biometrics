import os
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Constants
DATA_DIR = './png_txt'  # Define paths to data directories
TEST_SIZE = 0.25  # Value defines how much of the total dataset is used for validation
IMAGE_SIZE = (512, 512)  # Image dimensions


# Load Data Function
def load_data(data_dir):
    image_data = []
    metadata = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):  # Check if the item in root_dir is a directory
            print(f"Processing folder: {folder_name}")
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):
                    # Load image
                    img_path = os.path.join(folder_path, filename)
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize(IMAGE_SIZE)
                    img_array = np.array(img) / 255.0  # Normalize pixel values
                    image_data.append(img_array)
                    # Load corresponding metadata
                    txt_path = os.path.join(folder_path, filename[:-4] + '.txt')
                    with open(txt_path, 'r') as file:
                        meta = file.readlines()
                        metadata.append(meta[1].split(" ", )[1].strip())
        else:
            print(f"Error: Not a subdirectory: {folder_name}")
    return np.array(image_data), metadata


def calculate_eer(far, frr):
    eer = None
    for i in range(len(far)):
        if abs(far[i] - frr[i]) < 0.001:
            eer = (far[i] + frr[i]) / 2
            break
    return eer if eer is not None else (far[-1] + frr[-1]) / 2


def getStatistics(y_true, y_pred):
    thresholds = np.linspace(0, 1, 100)
    frr = []
    far = []
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        false_rejections = np.sum((y_pred_binary == 0) & (y_true == 1))
        false_accepts = np.sum((y_pred_binary == 1) & (y_true == 0))
        total_positives = np.sum(y_true == 1)
        total_negatives = np.sum(y_true == 0)
        frr.append(false_rejections / total_positives)
        far.append(false_accepts / total_negatives)
    # Calculate EER
    EER = calculate_eer(far, frr)
    FFR_avg = np.mean(frr)
    FRR_min = np.min(frr)
    FRR_max = np.max(frr)
    FAR_avg = np.mean(far)
    FAR_min = np.min(far)
    FAR_max = np.max(far)
    # Print results
    print(f'FRR Avg: {FFR_avg:.4f}, Min: {FRR_min:.4f}, Max: {FRR_max:.4f}')
    print(f'FAR Avg: {FAR_avg:.4f}, Min: {FAR_min:.4f}, Max: {FAR_max:.4f}')
    print(f'EER Avg: {EER:.4f}')


def main():
    # Load Data
    X, metadata = load_data(DATA_DIR)
    # Encode metadata (e.g., 'L', 'W', 'R', 'T', 'A') to numeric labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(metadata)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    train_count = len(X) * (1 - TEST_SIZE)
    test_count = len(X) * TEST_SIZE
    print("Data Set Includes " + str(len(X)) + " Features")
    print("Using " + str((1 - TEST_SIZE) * 100) + "% (" + str(train_count) + " Features) for Training")
    print("Using " + str(TEST_SIZE * 100) + "% (" + str(test_count) + " Features) for Testing")
    # Train the model using linear regression
    print("Training Model")
    model = LinearRegression()
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    # Save Model for Hybrid Testing
    with open('MSE.pkl', 'wb') as f:
        pickle.dump(model, f)
    # Predict on test set
    print("Testing Model")
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    # Calculate statistics
    getStatistics(y_test, y_pred)


if __name__ == '__main__':
    main()
