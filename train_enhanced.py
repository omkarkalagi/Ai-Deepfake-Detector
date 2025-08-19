import os
import zipfile
import urllib3
import requests
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.layers import LeakyReLU, GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json


class EnhancedDatasetHandler:
    """
    Enhanced class to handle dataset downloading, unzipping, loading, and processing with data augmentation.
    """

    def __init__(self, dataset_url, dataset_download_dir, dataset_file, dataset_dir, train_dir, test_dir, val_dir):
        """
        Initialize the EnhancedDatasetHandler with the specified parameters.
        """
        self.dataset_url = dataset_url
        self.dataset_download_dir = dataset_download_dir
        self.dataset_file = dataset_file
        self.dataset_dir = dataset_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

    def download_dataset(self):
        """Download the dataset from the specified URL."""
        if not os.path.exists(self.dataset_download_dir):
            os.makedirs(self.dataset_download_dir)
        file_path = os.path.join(self.dataset_download_dir, self.dataset_file)
        if os.path.exists(file_path):
            print(f'Dataset file {self.dataset_file} already exists at {file_path}')
            return True
        
        print(f'Downloading dataset from {self.dataset_url}...')
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(self.dataset_url, stream=True, verify=False)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file, tqdm(desc=self.dataset_file, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print(f'Dataset downloaded and saved to {file_path}')
        return True

    def unzip_dataset(self):
        """Unzip the downloaded dataset file."""
        file_path = os.path.join(self.dataset_download_dir, self.dataset_file)
        if os.path.exists(self.dataset_dir):
            print(f'Dataset is already extracted at {self.dataset_dir}')
            return True
        if not os.path.exists(file_path):
            print(f'Dataset file {file_path} not found')
            return False
        
        print(f'Extracting dataset...')
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_download_dir)
        print(f'Dataset extracted to {self.dataset_dir}')
        return True

    def get_image_dataset_from_directory(self, dir_name, is_training=False):
        """Load image dataset from the specified directory with optional augmentation."""
        dir_path = os.path.join(self.dataset_dir, dir_name)
        
        if is_training:
            # Data augmentation for training set
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
            return datagen.flow_from_directory(
                dir_path,
                target_size=(128, 128),
                batch_size=32,
                class_mode='binary',
                seed=42
            )
        else:
            # No augmentation for validation/test sets
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            return datagen.flow_from_directory(
                dir_path,
                target_size=(128, 128),
                batch_size=32,
                class_mode='binary',
                seed=42,
                shuffle=False
            )

    def load_split_data(self):
        """Load and split the dataset into training, validation, and test datasets."""
        train_data = self.get_image_dataset_from_directory(self.train_dir, is_training=True)
        test_data = self.get_image_dataset_from_directory(self.test_dir, is_training=False)
        val_data = self.get_image_dataset_from_directory(self.val_dir, is_training=False)
        return train_data, test_data, val_data


class EnhancedDeepfakeDetectorModel:
    """
    Enhanced class to create and train a deepfake detection model with improved architecture.
    """

    def __init__(self, input_shape=(128, 128, 3)):
        """Initialize the EnhancedDeepfakeDetectorModel."""
        self.input_shape = input_shape
        self.model = self._build_enhanced_model()

    def _build_enhanced_model(self):
        """Build an enhanced deepfake detection model architecture."""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling instead of Flatten
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(256, kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model

    def compile_model(self, learning_rate=0.001):
        """Compile the enhanced deepfake detection model."""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    def get_callbacks(self):
        """Get enhanced callbacks for training."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'deepfake_detector_model_enhanced.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks

    def train_model(self, train_data, val_data, epochs=100):
        """Train the enhanced deepfake detection model."""
        callbacks = self.get_callbacks()
        
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate_model(self, test_data):
        """Evaluate the enhanced deepfake detection model."""
        return self.model.evaluate(test_data, verbose=1)

    def save_model(self, path):
        """Save the enhanced deepfake detection model."""
        self.model.save(path)
        print(f'Model saved to {path}')

    def plot_training_history(self, history):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Plot training & validation loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Plot precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Plot recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


class EnhancedTrainModel:
    """Enhanced class to manage training of a deepfake detection model."""

    def __init__(self, dataset_url, dataset_download_dir, dataset_file, dataset_dir, train_dir, test_dir, val_dir):
        """Initialize the EnhancedTrainModel class."""
        self.dataset_handler = EnhancedDatasetHandler(
            dataset_url, dataset_download_dir, dataset_file, 
            dataset_dir, train_dir, test_dir, val_dir
        )

    def run_enhanced_training(self, learning_rate=0.001, epochs=100):
        """Run the enhanced training process."""
        print("Starting enhanced deepfake detection model training...")
        
        # Download and prepare dataset
        if not self.dataset_handler.download_dataset():
            print('Failed to download dataset')
            return
        if not self.dataset_handler.unzip_dataset():
            print('Failed to unzip dataset')
            return
        
        # Load data
        print("Loading dataset...")
        train_data, test_data, val_data = self.dataset_handler.load_split_data()
        
        # Create and compile model
        print("Creating enhanced model...")
        model = EnhancedDeepfakeDetectorModel()
        model.compile_model(learning_rate)
        
        # Print model summary
        print("\nModel Architecture:")
        model.model.summary()
        
        # Train model
        print("\nStarting training...")
        history = model.train_model(train_data, val_data, epochs)
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluation_metrics = model.evaluate_model(test_data)
        
        # Save model
        model.save_model('deepfake_detector_model_enhanced.keras')
        
        # Plot training history
        model.plot_training_history(history)
        
        # Save training results
        results = {
            'evaluation_metrics': evaluation_metrics,
            'final_accuracy': float(evaluation_metrics[1]),
            'final_precision': float(evaluation_metrics[2]),
            'final_recall': float(evaluation_metrics[3]),
            'epochs_trained': len(history.history['loss']),
            'learning_rate': learning_rate
        }
        
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Final Accuracy: {evaluation_metrics[1]:.4f}")
        print(f"Final Precision: {evaluation_metrics[2]:.4f}")
        print(f"Final Recall: {evaluation_metrics[3]:.4f}")
        
        return history, evaluation_metrics


if __name__ == '__main__':
    # Enhanced configuration
    dataset_url = 'https://www.kaggle.com/api/v1/datasets/download/manjilkarki/deepfake-and-real-images?datasetVersionNumber=1'
    dataset_download_dir = './data'
    dataset_file = 'dataset.zip'
    dataset_dir = './data/Dataset'
    train_dir = 'Train'
    test_dir = 'Test'
    val_dir = 'Validation'

    # Create enhanced trainer
    trainer = EnhancedTrainModel(
        dataset_url=dataset_url,
        dataset_download_dir=dataset_download_dir,
        dataset_file=dataset_file,
        dataset_dir=dataset_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        val_dir=val_dir
    )

    # Run enhanced training
    history, evaluation_metrics = trainer.run_enhanced_training(
        learning_rate=0.001, 
        epochs=100
    )
