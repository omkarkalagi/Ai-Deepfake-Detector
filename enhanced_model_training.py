#!/usr/bin/env python3
"""
Enhanced Deepfake Detection Model Training
Based on latest research and best practices from 2024
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDeepfakeDetector:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def create_advanced_model(self):
        """
        Create an advanced deepfake detection model using:
        1. EfficientNet backbone for feature extraction
        2. Multi-scale feature fusion
        3. Attention mechanisms
        4. Binary classification head
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation layers (built into model for consistency)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        
        # Backbone: EfficientNetB0 (pre-trained on ImageNet)
        backbone = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_tensor=x
        )
        
        # Fine-tune last few layers
        for layer in backbone.layers[:-20]:
            layer.trainable = False
            
        # Multi-scale feature extraction
        # Get features from different scales
        feature_maps = []
        for i, layer in enumerate(backbone.layers):
            if 'block' in layer.name and 'add' in layer.name:
                feature_maps.append(layer.output)
        
        # Use the last feature map as primary
        x = backbone.output
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Attention mechanism
        attention = layers.Dense(x.shape[-1], activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        
        # Feature enhancement layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Classification head
        predictions = layers.Dense(1, activation='sigmoid', name='predictions')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=predictions)
        
        return model
    
    def compile_model(self, learning_rate=1e-4):
        """Compile the model with advanced optimization"""
        if self.model is None:
            self.model = self.create_advanced_model()
            
        # Use Adam optimizer with learning rate scheduling
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Compile with binary crossentropy and multiple metrics
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info("Model compiled successfully")
        return self.model
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create enhanced data generators with augmentation"""
        
        # Training data generator with heavy augmentation
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
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def create_callbacks(self, model_name='enhanced_deepfake_detector'):
        """Create training callbacks for better training"""
        
        callbacks_list = [
            # Model checkpointing
            callbacks.ModelCheckpoint(
                filepath=f'{model_name}_best.keras',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # CSV logging
            callbacks.CSVLogger(f'{model_name}_training_log.csv'),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks_list
    
    def train_model(self, train_generator, val_generator, epochs=50):
        """Train the model with advanced techniques"""
        
        if self.model is None:
            self.compile_model()
        
        # Get callbacks
        callbacks_list = self.create_callbacks()
        
        # Calculate steps
        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history
    
    def evaluate_model(self, test_generator):
        """Comprehensive model evaluation"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Evaluate on test set
        test_loss, test_acc, test_precision, test_recall, test_auc = self.model.evaluate(
            test_generator, verbose=1
        )
        
        # Generate predictions
        predictions = self.model.predict(test_generator)
        y_pred = (predictions > 0.5).astype(int)
        y_true = test_generator.classes
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"Test Results:")
        logger.info(f"Accuracy: {test_acc:.4f}")
        logger.info(f"Precision: {test_precision:.4f}")
        logger.info(f"Recall: {test_recall:.4f}")
        logger.info(f"AUC: {test_auc:.4f}")
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # AUC
        axes[1, 0].plot(self.history.history['auc'], label='Training AUC')
        axes[1, 0].plot(self.history.history['val_auc'], label='Validation AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        
        # Precision & Recall
        axes[1, 1].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 1].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='enhanced_deepfake_detector_final.keras'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save model configuration
        config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'model_architecture': 'EfficientNetB0_Enhanced',
            'training_date': datetime.now().isoformat(),
            'framework': 'TensorFlow/Keras'
        }
        
        config_path = filepath.replace('.keras', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model configuration saved to {config_path}")

def download_datasets():
    """
    Download and prepare datasets for training
    Based on research: FaceForensics++, CelebDF, DFDC
    """
    
    logger.info("Dataset download instructions:")
    logger.info("1. FaceForensics++: https://github.com/ondyari/FaceForensics")
    logger.info("2. CelebDF: https://github.com/yuezunli/celeb-deepfakeforensics")
    logger.info("3. DFDC: https://www.kaggle.com/c/deepfake-detection-challenge")
    
    # Create directory structure
    os.makedirs('datasets/train/real', exist_ok=True)
    os.makedirs('datasets/train/fake', exist_ok=True)
    os.makedirs('datasets/val/real', exist_ok=True)
    os.makedirs('datasets/val/fake', exist_ok=True)
    os.makedirs('datasets/test/real', exist_ok=True)
    os.makedirs('datasets/test/fake', exist_ok=True)
    
    logger.info("Dataset directories created. Please populate with downloaded data.")

def main():
    """Main training pipeline"""
    
    # Initialize detector
    detector = EnhancedDeepfakeDetector(input_shape=(224, 224, 3))
    
    # Compile model
    model = detector.compile_model(learning_rate=1e-4)
    
    # Print model summary
    model.summary()
    
    # Check if datasets exist
    if not os.path.exists('datasets/train'):
        logger.info("Datasets not found. Creating directory structure...")
        download_datasets()
        logger.info("Please download and organize datasets before training.")
        return
    
    # Create data generators
    train_gen, val_gen = detector.create_data_generators(
        'datasets/train',
        'datasets/val',
        batch_size=32
    )
    
    # Train model
    history = detector.train_model(train_gen, val_gen, epochs=50)
    
    # Plot training history
    detector.plot_training_history()
    
    # Evaluate if test data exists
    if os.path.exists('datasets/test'):
        test_gen = detector.create_data_generators(
            'datasets/test',
            'datasets/test',  # Same directory for test
            batch_size=32
        )[0]  # Only need one generator
        
        results = detector.evaluate_model(test_gen)
        
        # Save results
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Save final model
    detector.save_model('enhanced_deepfake_detector_final.keras')
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
