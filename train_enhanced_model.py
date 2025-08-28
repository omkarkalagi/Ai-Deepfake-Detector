"""
Training script for enhanced deepfake detection model
Integrates multiple datasets and advanced training techniques
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import requests
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDeepfakeTrainer:
    def __init__(self, model_name="enhanced_deepfake_detector_v2"):
        self.model_name = model_name
        self.model = None
        self.img_size = (224, 224)
        self.batch_size = 32
        self.num_classes = 3  # real, fake, edited
        self.class_names = ['real', 'fake', 'edited']
        
        # Create directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for training"""
        dirs = [
            'models',
            'data/train/real',
            'data/train/fake', 
            'data/train/edited',
            'data/val/real',
            'data/val/fake',
            'data/val/edited',
            'logs',
            'checkpoints'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        logger.info("Directory structure created")
    
    def download_sample_data(self):
        """Download sample training data"""
        logger.info("Setting up sample training data...")
        
        # Create sample data info files
        sample_info = {
            'datasets': [
                {
                    'name': 'FaceForensics++',
                    'url': 'https://github.com/ondyari/FaceForensics',
                    'description': 'Contains Deepfakes, Face2Face, FaceSwap, NeuralTextures',
                    'size': '~500GB',
                    'samples': '1.8M frames'
                },
                {
                    'name': 'CelebDF',
                    'url': 'https://github.com/yuezunli/celeb-deepfakeforensics',
                    'description': 'High-quality celebrity deepfakes',
                    'size': '~15GB', 
                    'samples': '590 real + 5,639 fake videos'
                },
                {
                    'name': 'DFDC',
                    'url': 'https://ai.facebook.com/datasets/dfdc/',
                    'description': 'Facebook Deepfake Detection Challenge dataset',
                    'size': '~470GB',
                    'samples': '100K+ videos'
                }
            ],
            'instructions': 'Download datasets manually and place in appropriate directories',
            'preprocessing': 'Extract frames, resize to 224x224, augment data'
        }
        
        with open('data/dataset_info.json', 'w') as f:
            json.dump(sample_info, f, indent=2)
            
        # Create sample placeholder files
        for split in ['train', 'val']:
            for class_name in self.class_names:
                readme_path = f'data/{split}/{class_name}/README.txt'
                with open(readme_path, 'w') as f:
                    f.write(f"Place {class_name} images here for {split} set\n")
                    f.write(f"Recommended: 10,000+ images per class for training\n")
                    f.write(f"Image format: JPG/PNG, Size: 224x224 or larger\n")
        
        logger.info("Sample data structure created. Please add actual training images.")
    
    def create_enhanced_model(self):
        """Create enhanced model architecture"""
        logger.info("Creating enhanced model architecture...")
        
        # Input layer
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        
        # Preprocessing
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # Base model - EfficientNetB4
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_tensor=x
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers with batch normalization
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layers
        # Main classification output
        main_output = layers.Dense(self.num_classes, activation='softmax', name='classification')(x)
        
        # Confidence output
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=[main_output, confidence_output])
        
        # Compile model
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss={
                'classification': 'categorical_crossentropy',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={'classification': 1.0, 'confidence': 0.3},
            metrics={
                'classification': ['accuracy', 'precision', 'recall'],
                'confidence': ['mae']
            }
        )
        
        self.model = model
        logger.info(f"Model created with {model.count_params():,} parameters")
        return model
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        logger.info("Creating data generators...")
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        try:
            train_generator = train_datagen.flow_from_directory(
                'data/train',
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=True
            )
            
            val_generator = val_datagen.flow_from_directory(
                'data/val',
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=False
            )
            
            logger.info(f"Training samples: {train_generator.samples}")
            logger.info(f"Validation samples: {val_generator.samples}")
            
            return train_generator, val_generator
            
        except Exception as e:
            logger.error(f"Error creating data generators: {e}")
            logger.info("Please ensure training data is properly organized in data/train and data/val directories")
            return None, None
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_classification_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                f'checkpoints/{self.model_name}_best.h5',
                monitor='val_classification_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                f'logs/{self.model_name}_training_log.csv',
                append=True
            ),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir=f'logs/tensorboard/{self.model_name}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks_list
    
    def train_model(self, epochs=100):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        
        if train_gen is None or val_gen is None:
            logger.error("Cannot proceed without training data")
            return None
        
        if train_gen.samples == 0 or val_gen.samples == 0:
            logger.error("No training samples found. Please add images to data directories.")
            return None
        
        # Setup callbacks
        callback_list = self.setup_callbacks()
        
        # Phase 1: Train with frozen base model
        logger.info("Phase 1: Training with frozen base model...")
        history1 = self.model.fit(
            train_gen,
            epochs=epochs//3,
            validation_data=val_gen,
            callbacks=callback_list,
            verbose=1
        )
        
        # Phase 2: Fine-tuning with unfrozen base model
        logger.info("Phase 2: Fine-tuning with unfrozen base model...")
        
        # Unfreeze base model
        self.model.get_layer('efficientnetb4').trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001),
            loss={
                'classification': 'categorical_crossentropy',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={'classification': 1.0, 'confidence': 0.3},
            metrics={
                'classification': ['accuracy', 'precision', 'recall'],
                'confidence': ['mae']
            }
        )
        
        # Continue training
        history2 = self.model.fit(
            train_gen,
            epochs=epochs,
            initial_epoch=len(history1.history['loss']),
            validation_data=val_gen,
            callbacks=callback_list,
            verbose=1
        )
        
        # Combine histories
        combined_history = {}
        for key in history1.history.keys():
            combined_history[key] = history1.history[key] + history2.history[key]
        
        # Save final model
        self.save_model()
        
        # Plot training history
        self.plot_training_history(combined_history)
        
        logger.info("Training completed successfully!")
        return combined_history
    
    def save_model(self):
        """Save the trained model"""
        model_path = f'models/{self.model_name}.h5'
        self.model.save(model_path)
        
        # Save model configuration
        config = {
            'model_name': self.model_name,
            'architecture': 'EfficientNetB4 + Enhanced Dense Layers',
            'input_size': self.img_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'created_at': datetime.now().isoformat(),
            'framework': 'TensorFlow/Keras',
            'version': '2.1'
        }
        
        config_path = f'models/{self.model_name}_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Config saved: {config_path}")
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history['classification_accuracy'], label='Training')
        axes[0, 0].plot(history['val_classification_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Training')
        axes[0, 1].plot(history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history['classification_precision'], label='Training')
        axes[1, 0].plot(history['val_classification_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history['classification_recall'], label='Training')
        axes[1, 1].plot(history['val_classification_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'models/{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Training history plot saved: models/{self.model_name}_training_history.png")

def main():
    """Main training function"""
    print("=" * 60)
    print("Enhanced Deepfake Detection Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = EnhancedDeepfakeTrainer()
    
    # Setup sample data structure
    trainer.download_sample_data()
    
    # Create model
    model = trainer.create_enhanced_model()
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Check for training data
    train_path = Path('data/train')
    if not any(train_path.rglob('*.jpg')) and not any(train_path.rglob('*.png')):
        print("\n" + "="*60)
        print("TRAINING DATA REQUIRED")
        print("="*60)
        print("Please add training images to the following directories:")
        print("- data/train/real/     (Real person images)")
        print("- data/train/fake/     (Deepfake images)")
        print("- data/train/edited/   (Edited/manipulated images)")
        print("- data/val/real/       (Validation real images)")
        print("- data/val/fake/       (Validation fake images)")
        print("- data/val/edited/     (Validation edited images)")
        print("\nRecommended datasets:")
        print("1. FaceForensics++: https://github.com/ondyari/FaceForensics")
        print("2. CelebDF: https://github.com/yuezunli/celeb-deepfakeforensics")
        print("3. DFDC: https://ai.facebook.com/datasets/dfdc/")
        print("\nMinimum 1,000 images per class recommended for meaningful training.")
        print("="*60)
        return
    
    # Start training
    try:
        history = trainer.train_model(epochs=50)  # Reduced for demo
        
        if history:
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Model saved: models/{trainer.model_name}.h5")
            print(f"Config saved: models/{trainer.model_name}_config.json")
            print(f"Training logs: logs/{trainer.model_name}_training_log.csv")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nTraining failed: {e}")

if __name__ == "__main__":
    main()
