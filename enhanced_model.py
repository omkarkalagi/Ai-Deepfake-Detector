"""
Enhanced Deepfake Detection Model with Multiple Dataset Support
Supports FaceForensics++, DFDC, CelebDF, and other comprehensive datasets
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB4, ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import json
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDeepfakeDetector:
    def __init__(self, model_name="enhanced_deepfake_detector"):
        self.model_name = model_name
        self.model = None
        self.history = None
        self.class_names = ['real', 'fake', 'edited']
        self.img_size = (224, 224)
        self.batch_size = 32
        
    def download_datasets(self):
        """Download and prepare multiple datasets"""
        datasets = {
            'faceforensics': {
                'url': 'https://github.com/ondyari/FaceForensics/releases/download/v1.0/FaceForensics++.zip',
                'description': 'FaceForensics++ dataset with multiple manipulation methods'
            },
            'celeb_df': {
                'url': 'https://drive.google.com/file/d/1VIOYB-kzDVGed2eRHjggBLnzJbQHwM5B/view',
                'description': 'CelebDF dataset with celebrity deepfakes'
            }
        }
        
        logger.info("Dataset download URLs prepared. Manual download required for large datasets.")
        return datasets
    
    def create_enhanced_model(self):
        """Create an enhanced model architecture"""
        # Base model with EfficientNet
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Enhanced architecture
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        
        # Preprocessing
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global pooling and regularization
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers with batch normalization
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Multi-class output (real, fake, edited)
        outputs = layers.Dense(3, activation='softmax', name='classification')(x)
        
        # Confidence output
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(x)
        
        model = tf.keras.Model(inputs, [outputs, confidence_output])
        
        # Compile with multiple losses
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=0.001),
            loss={
                'classification': 'categorical_crossentropy',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={'classification': 1.0, 'confidence': 0.5},
            metrics={
                'classification': ['accuracy', 'precision', 'recall'],
                'confidence': ['mae']
            }
        )
        
        self.model = model
        return model
    
    def create_data_generators(self, train_dir, val_dir):
        """Create enhanced data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names
        )
        
        return train_generator, val_generator
    
    def train_model(self, train_generator, val_generator, epochs=50):
        """Train the model with callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'models/{self.model_name}_best.h5',
                monitor='val_classification_accuracy',
                save_best_only=True,
                save_weights_only=False
            ),
            tf.keras.callbacks.CSVLogger(
                f'models/{self.model_name}_training_log.csv'
            )
        ]
        
        # Initial training
        logger.info("Starting initial training phase...")
        history1 = self.model.fit(
            train_generator,
            epochs=epochs//2,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        logger.info("Starting fine-tuning phase...")
        self.model.get_layer('efficientnetb4').trainable = True
        
        # Lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.AdamW(learning_rate=0.0001),
            loss={
                'classification': 'categorical_crossentropy',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={'classification': 1.0, 'confidence': 0.5},
            metrics={
                'classification': ['accuracy', 'precision', 'recall'],
                'confidence': ['mae']
            }
        )
        
        history2 = self.model.fit(
            train_generator,
            epochs=epochs//2,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'classification_accuracy': history1.history['classification_accuracy'] + history2.history['classification_accuracy'],
            'val_classification_accuracy': history1.history['val_classification_accuracy'] + history2.history['val_classification_accuracy']
        }
        
        return self.history
    
    def predict_image(self, image_path):
        """Enhanced prediction with confidence scores"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0
        
        # Make prediction
        predictions, confidence = self.model.predict(img, verbose=0)
        
        # Get class probabilities
        class_probs = predictions[0]
        predicted_class_idx = np.argmax(class_probs)
        predicted_class = self.class_names[predicted_class_idx]
        
        # Calculate percentages
        real_percentage = float(class_probs[0] * 100)
        fake_percentage = float(class_probs[1] * 100)
        edited_percentage = float(class_probs[2] * 100)
        
        # Overall confidence
        overall_confidence = float(confidence[0][0] * 100)
        
        return {
            'prediction': predicted_class,
            'real_percentage': round(real_percentage, 2),
            'fake_percentage': round(fake_percentage, 2),
            'edited_percentage': round(edited_percentage, 2),
            'confidence': round(overall_confidence, 2),
            'class_probabilities': {
                'real': round(real_percentage, 2),
                'fake': round(fake_percentage, 2),
                'edited': round(edited_percentage, 2)
            }
        }
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f'models/{self.model_name}.h5'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
        # Save model configuration
        config = {
            'model_name': self.model_name,
            'class_names': self.class_names,
            'img_size': self.img_size,
            'created_at': datetime.now().isoformat(),
            'architecture': 'EfficientNetB4 + Enhanced Dense Layers'
        }
        
        config_path = filepath.replace('.h5', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        
        # Load configuration if available
        config_path = filepath.replace('.h5', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.class_names = config.get('class_names', self.class_names)
                self.img_size = tuple(config.get('img_size', self.img_size))
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history['classification_accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history['val_classification_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'models/{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_demo_data():
    """Create demo training data structure"""
    demo_structure = {
        'data/train/real': 'Real images from various sources',
        'data/train/fake': 'Deepfake images from FaceForensics++, DFDC',
        'data/train/edited': 'Edited/manipulated images',
        'data/val/real': 'Validation real images',
        'data/val/fake': 'Validation fake images', 
        'data/val/edited': 'Validation edited images'
    }
    
    for path, description in demo_structure.items():
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'README.txt'), 'w') as f:
            f.write(f"{description}\n")
            f.write("Place your training images in this directory.\n")
    
    logger.info("Demo data structure created in 'data/' directory")

if __name__ == "__main__":
    # Initialize detector
    detector = EnhancedDeepfakeDetector()
    
    # Create model
    model = detector.create_enhanced_model()
    print(f"Model created with {model.count_params():,} parameters")
    
    # Create demo data structure
    create_demo_data()
    
    # Print dataset information
    datasets = detector.download_datasets()
    print("\nRecommended Datasets:")
    for name, info in datasets.items():
        print(f"- {name}: {info['description']}")
    
    print(f"\nModel architecture summary:")
    model.summary()
