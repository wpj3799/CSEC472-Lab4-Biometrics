import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Constants
IMAGE_SIZE = (512, 512)  # Image dimensions
EPOCHS = 20
NUM_CLASSES = 5  # Number of fingerprint types (L, W, R, T, A)
TEST_SIZE = 0.25  # Value defines how much of the total dataset is used for validation
DATA_DIR = './png_txt'  # Define paths to data directories
checkpoint_path = "./cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)


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


# Create CCN Neural Model
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


# Evaluate the Testing Data on the Trained CNN Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    # Calculate False Rejection Rate (FRR) and False Acceptance Rate (FAR)
    FRR = np.zeros(NUM_CLASSES)
    FAR = np.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        # Calculate FRR for class i
        total_rejections = np.sum(conf_matrix[i]) - conf_matrix[i, i]
        FRR[i] = total_rejections / np.sum(conf_matrix[i])
        # Calculate FAR for class i
        total_acceptances = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        FAR[i] = total_acceptances / (np.sum(conf_matrix) - np.sum(conf_matrix[i]))
    # Calculate average, minimum, and maximum FRR and FAR
    FRR_avg = np.mean(FRR)
    FRR_min = np.min(FRR)
    FRR_max = np.max(FRR)
    FAR_avg = np.mean(FAR)
    FAR_min = np.min(FAR)
    FAR_max = np.max(FAR)
    # Calculate Equal Error Rate (EER)
    EER = (FRR + FAR) / 2
    EER_avg = np.mean(EER)
    # Print results
    print(f'FRR Avg: {FRR_avg:.4f}, Min: {FRR_min:.4f}, Max: {FRR_max:.4f}')
    print(f'FAR Avg: {FAR_avg:.4f}, Min: {FAR_min:.4f}, Max: {FAR_max:.4f}')
    print(f'EER Avg: {EER_avg:.4f}')


def main():
    # Load Data
    X, metadata = load_data(DATA_DIR)
    # Encode metadata (e.g., 'L', 'W', 'R', 'T', 'A') to numeric labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(metadata)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    train_count = len(X) * (1 - TEST_SIZE)
    test_count = len(X) * TEST_SIZE
    print("Data Set Includes " + str(len(X)) + " Features")
    print("Using " + str((1 - TEST_SIZE) * 100) + "% (" + str(train_count) + " Features) for Training")
    print("Using " + str(TEST_SIZE * 100) + "% (" + str(test_count) + " Features) for Testing")
    # Instantiate model
    model = create_model()
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Load the previously saved weights IF NEEDED
    model.load_weights(latest)
    # # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=5*75)
    # # Save the weights using the `checkpoint_path` format
    # model.save_weights(checkpoint_path.format(epoch=0))
    # Train model on training data
    print("Training Model")
    model.fit(X_train, y_train, epochs=1, validation_split=0.2, callbacks=[cp_callback])
    # Save Model for Hybrid Testing
    model.save("./cnn_model.keras")
    # Evaluate model on testing data
    print("Testing Model")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
