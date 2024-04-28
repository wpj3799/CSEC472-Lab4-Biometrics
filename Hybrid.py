import os
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from threading import Thread
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


# Run CNN Model
def run_cnn(model, X_test, y_true, results):
    print("Testing CNN Model")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    FRR = np.zeros(5)
    FAR = np.zeros(5)
    for i in range(5):
        total_rejections = np.sum(conf_matrix[i]) - conf_matrix[i, i]
        FRR[i] = total_rejections / np.sum(conf_matrix[i])
        total_acceptances = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        FAR[i] = total_acceptances / (np.sum(conf_matrix) - np.sum(conf_matrix[i]))
    FRR_avg = np.mean(FRR)
    FRR_min = np.min(FRR)
    FRR_max = np.max(FRR)
    FAR_avg = np.mean(FAR)
    FAR_min = np.min(FAR)
    FAR_max = np.max(FAR)
    EER = (FRR + FAR) / 2
    EER_avg = np.mean(EER)
    results["cnn"] = [FRR_avg, FRR_min, FRR_max, FAR_avg, FAR_min, FAR_max, EER_avg]
    print("Testing CNN Model Complete")


# Run MSE Model
def run_mse(model, X_test, y_test, results):
    print("Testing MSE Model")
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    thresholds = np.linspace(0, 1, 100)
    frr = []
    far = []
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        false_rejections = np.sum((y_pred_binary == 0) & (y_test == 1))
        false_accepts = np.sum((y_pred_binary == 1) & (y_test == 0))
        total_positives = np.sum(y_test == 1)
        total_negatives = np.sum(y_test == 0)
        frr.append(false_rejections / total_positives)
        far.append(false_accepts / total_negatives)
    EER = calculate_eer(far, frr)
    FRR_avg = np.mean(frr)
    FRR_min = np.min(frr)
    FRR_max = np.max(frr)
    FAR_avg = np.mean(far)
    FAR_min = np.min(far)
    FAR_max = np.max(far)
    EER_avg = np.mean(EER)
    results["mse"] = [FRR_avg, FRR_min, FRR_max, FAR_avg, FAR_min, FAR_max, EER_avg]
    print("Testing MSE Model Complete")


# Run SVM Model
def run_svm(model, X_test, y_test, results):
    print("Testing SVM Model")
    y_pred = model.predict(X_test.reshape(len(X_test), -1))
    frr = 1 - accuracy_score(y_test, y_pred, normalize=True)
    far = 1 - frr
    EER = (frr + far) / 2
    FRR_avg = np.mean(frr)
    FRR_min = np.min(frr)
    FRR_max = np.max(frr)
    FAR_avg = np.mean(far)
    FAR_min = np.min(far)
    FAR_max = np.max(far)
    EER_avg = np.mean(EER)
    results["svm"] = [FRR_avg, FRR_min, FRR_max, FAR_avg, FAR_min, FAR_max, EER_avg]
    print("Testing SVM Model Complete")


def main():
    # Load Data
    X, metadata = load_data(DATA_DIR)
    # Encode metadata (e.g., 'L', 'W', 'R', 'T', 'A') to numeric labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(metadata)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    # Load models
    cnn_model = load_model("CNN.keras")
    with open("MSE.pkl", "rb") as f:
        mse_model = pickle.load(f)
    with open("SVM.pkl", "rb") as f:
        svm_model = pickle.load(f)
    # Create threads to run models concurrently
    results = {"cnn": [], "mse": [], "svm": []}
    # Create a thread for each model
    cnn_thread = Thread(target=run_cnn, args=(cnn_model, X_test, y_test, results))
    mse_thread = Thread(target=run_mse, args=(mse_model, X_test, y_test, results))
    svm_thread = Thread(target=run_svm, args=(svm_model, X_test, y_test, results))
    # Start threads
    cnn_thread.start()
    mse_thread.start()
    svm_thread.start()
    # Join threads
    cnn_thread.join()
    mse_thread.join()
    svm_thread.join()
    # Average results
    cnn_values = results["cnn"]
    mse_values = results["mse"]
    svm_values = results["svm"]
    # Calculate averages
    FFR_avg = np.mean([cnn_values[0], mse_values[0], svm_values[0]])
    FRR_min = np.mean([cnn_values[1], mse_values[1], svm_values[1]])
    FRR_max = np.mean([cnn_values[2], mse_values[2], svm_values[2]])
    FAR_avg = np.mean([cnn_values[3], mse_values[3], svm_values[3]])
    FAR_min = np.mean([cnn_values[4], mse_values[4], svm_values[4]])
    FAR_max = np.mean([cnn_values[5], mse_values[5], svm_values[5]])
    EER = np.mean([cnn_values[6], mse_values[6], svm_values[6]])
    print(f'FRR Avg: {FFR_avg:.4f}, Min: {FRR_min:.4f}, Max: {FRR_max:.4f}')
    print(f'FAR Avg: {FAR_avg:.4f}, Min: {FAR_min:.4f}, Max: {FAR_max:.4f}')
    print(f'EER Avg: {EER:.4f}')


if __name__ == "__main__":
    main()
