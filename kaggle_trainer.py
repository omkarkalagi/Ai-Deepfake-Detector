#!/usr/bin/env python3
"""
Advanced Kaggle Dataset Trainer for Deepfake Detection
Downloads and trains with real Kaggle datasets for maximum accuracy.
"""

import os
import sys
import json
import zipfile
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime


class KaggleDatasetTrainer:
    """Advanced trainer using real Kaggle datasets for deepfake detection."""
    
    def __init__(self, data_dir="kaggle_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model = None
        self.history = None
        
        # Kaggle datasets for deepfake detection
        self.datasets = {
            "deepfake-and-real-images": {
                "url": "https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images",
                "description": "Real vs Deepfake Images Dataset",
                "size": "~500MB",
                "samples": "2000+ images"
            },
            "real-and-fake-face-detection": {
                "url": "https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection", 
                "description": "Real and Fake Face Detection Dataset",
                "size": "~1.2GB",
                "samples": "4000+ images"
            },
            "140k-real-and-fake-faces": {
                "url": "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces",
                "description": "140K Real and Fake Faces",
                "size": "~2.8GB", 
                "samples": "140,000+ images"
            }
        }
    
    def setup_kaggle_api(self):
        """Setup Kaggle API credentials."""
        print("🔑 Setting up Kaggle API...")
        
        # Check if kaggle.json exists
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            print("⚠️  Kaggle API credentials not found!")
            print("📝 To use Kaggle datasets:")
            print("   1. Go to https://www.kaggle.com/account")
            print("   2. Click 'Create New API Token'")
            print("   3. Save kaggle.json to ~/.kaggle/")
            print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        try:
            import kaggle
            print("✅ Kaggle API configured successfully")
            return True
        except ImportError:
            print("❌ Kaggle package not installed. Run: pip install kaggle")
            return False
    
    def download_kaggle_dataset(self, dataset_name):
        """Download a specific Kaggle dataset."""
        if not self.setup_kaggle_api():
            return False
        
        try:
            import kaggle
            
            print(f"📥 Downloading {dataset_name}...")
            dataset_path = self.data_dir / dataset_name
            
            if dataset_path.exists():
                print(f"✅ Dataset {dataset_name} already exists")
                return True
            
            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(dataset_path),
                unzip=True
            )
            
            print(f"✅ Downloaded {dataset_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
            return False
    
    def prepare_dataset(self, dataset_path):
        """Prepare dataset for training."""
        print("📊 Preparing dataset for training...")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            return None, None, None, None
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        labels = []
        
        # Look for organized folder structure (real/fake or similar)
        real_folders = ['real', 'authentic', 'original', 'true']
        fake_folders = ['fake', 'deepfake', 'synthetic', 'generated', 'false']
        
        for folder in dataset_path.iterdir():
            if folder.is_dir():
                folder_name = folder.name.lower()
                
                # Determine label based on folder name
                if any(real_word in folder_name for real_word in real_folders):
                    label = 0  # Real
                elif any(fake_word in folder_name for fake_word in fake_folders):
                    label = 1  # Fake
                else:
                    continue  # Skip unknown folders
                
                # Collect images from this folder
                for img_file in folder.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        image_files.append(str(img_file))
                        labels.append(label)
        
        if len(image_files) == 0:
            print("❌ No images found in dataset")
            return None, None, None, None
        
        print(f"✅ Found {len(image_files)} images")
        print(f"   - Real images: {labels.count(0)}")
        print(f"   - Fake images: {labels.count(1)}")
        
        # Split dataset
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_files, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Dataset split:")
        print(f"   - Training: {len(X_train)} images")
        print(f"   - Validation: {len(X_val)} images") 
        print(f"   - Testing: {len(X_test)} images")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), len(image_files)
    
    def create_advanced_model(self, input_shape=(224, 224, 3)):
        """Create an advanced model for high accuracy."""
        print("🏗️  Creating advanced deepfake detection model...")
        
        # Use EfficientNetB0 as base model
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(), 
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        print("✅ Advanced model created")
        print(f"   - Total parameters: {model.count_params():,}")
        
        return model
    
    def create_data_generators(self, train_data, val_data, batch_size=32, input_size=(224, 224)):
        """Create data generators with advanced augmentation."""
        print("🔄 Creating data generators...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Advanced data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        # Simple rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        def load_and_preprocess_image(image_path, target_size):
            """Load and preprocess image."""
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize(target_size)
                return np.array(img)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                return np.zeros((*target_size, 3), dtype=np.uint8)
        
        def data_generator(image_paths, labels, datagen, batch_size, target_size):
            """Custom data generator."""
            while True:
                indices = np.random.permutation(len(image_paths))
                
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_images = []
                    batch_labels = []
                    
                    for idx in batch_indices:
                        img = load_and_preprocess_image(image_paths[idx], target_size)
                        batch_images.append(img)
                        batch_labels.append(labels[idx])
                    
                    batch_images = np.array(batch_images)
                    batch_labels = np.array(batch_labels)
                    
                    # Apply augmentation
                    for j in range(len(batch_images)):
                        batch_images[j] = datagen.random_transform(batch_images[j])
                        batch_images[j] = datagen.standardize(batch_images[j])
                    
                    yield batch_images, batch_labels
        
        train_generator = data_generator(X_train, y_train, train_datagen, batch_size, input_size)
        val_generator = data_generator(X_val, y_val, val_datagen, batch_size, input_size)
        
        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // batch_size
        
        print(f"✅ Data generators created")
        print(f"   - Steps per epoch: {steps_per_epoch}")
        print(f"   - Validation steps: {validation_steps}")
        
        return train_generator, val_generator, steps_per_epoch, validation_steps
    
    def get_callbacks(self, model_name="kaggle_deepfake_model.keras"):
        """Get training callbacks."""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                model_name,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            callbacks.CSVLogger(
                'training_log.csv',
                append=True
            )
        ]
        
        return callbacks_list
    
    def train_model(self, train_generator, val_generator, steps_per_epoch, validation_steps, epochs=50):
        """Train the model with Kaggle data."""
        print(f"🏋️  Starting training for {epochs} epochs...")
        
        # Get callbacks
        callback_list = self.get_callbacks()
        
        # Train the model
        start_time = time.time()
        
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print("✅ Training completed")
        print(f"   - Training time: {training_time/3600:.2f} hours")
        print(f"   - Final accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"   - Final val accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        
        return self.history
    
    def fine_tune_model(self, train_generator, val_generator, steps_per_epoch, validation_steps, epochs=20):
        """Fine-tune the model by unfreezing base layers."""
        print("🔧 Starting fine-tuning...")
        
        # Unfreeze the base model
        self.model.layers[0].trainable = True
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001/10),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Fine-tune with fewer epochs
        fine_tune_history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=self.get_callbacks("kaggle_deepfake_model_finetuned.keras"),
            verbose=1
        )
        
        print("✅ Fine-tuning completed")
        return fine_tune_history
    
    def evaluate_model(self, test_data, batch_size=32):
        """Evaluate the model on test data."""
        print("📊 Evaluating model on test data...")
        
        X_test, y_test = test_data
        
        # Create test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        def load_test_images(image_paths, labels, target_size=(224, 224)):
            """Load test images."""
            images = []
            valid_labels = []
            
            for i, path in enumerate(image_paths):
                try:
                    img = Image.open(path).convert('RGB')
                    img = img.resize(target_size)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    valid_labels.append(labels[i])
                except Exception as e:
                    print(f"Error loading test image {path}: {e}")
                    continue
            
            return np.array(images), np.array(valid_labels)
        
        X_test_processed, y_test_processed = load_test_images(X_test, y_test)
        
        # Make predictions
        predictions = self.model.predict(X_test_processed, batch_size=batch_size, verbose=1)
        y_pred = (predictions > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test_processed)
        
        # Classification report
        report = classification_report(y_test_processed, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_processed, y_pred)
        
        results = {
            'accuracy': accuracy * 100,
            'precision': report['1']['precision'] * 100,
            'recall': report['1']['recall'] * 100,
            'f1_score': report['1']['f1-score'] * 100,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'test_samples': len(y_test_processed)
        }
        
        print(f"✅ Model Evaluation Results:")
        print(f"   - Test Accuracy: {results['accuracy']:.2f}%")
        print(f"   - Precision: {results['precision']:.2f}%")
        print(f"   - Recall: {results['recall']:.2f}%")
        print(f"   - F1-Score: {results['f1_score']:.2f}%")
        print(f"   - Test samples: {results['test_samples']}")
        
        return results
    
    def save_model(self, filename="kaggle_deepfake_detector.keras"):
        """Save the trained model."""
        if self.model:
            self.model.save(filename)
            print(f"💾 Model saved as {filename}")
            
            # Save model info
            model_info = {
                'filename': filename,
                'created': datetime.now().isoformat(),
                'architecture': 'EfficientNetB0 + Custom Head',
                'input_shape': [224, 224, 3],
                'parameters': int(self.model.count_params()),
                'training_completed': True
            }
            
            with open(filename.replace('.keras', '_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"📄 Model info saved")
        else:
            print("❌ No model to save")


def main():
    """Main training function."""
    print("🚀 Kaggle Dataset Deepfake Trainer")
    print("=" * 60)
    
    trainer = KaggleDatasetTrainer()
    
    # Download datasets
    print("\n📥 Downloading Kaggle datasets...")
    datasets_to_use = [
        "manjilkarki/deepfake-and-real-images",
        # Add more datasets as needed
    ]
    
    for dataset in datasets_to_use:
        if trainer.download_kaggle_dataset(dataset):
            print(f"✅ {dataset} ready for training")
        else:
            print(f"⚠️  Skipping {dataset} - download failed")
    
    # Use the first available dataset for training
    dataset_path = trainer.data_dir / "manjilkarki" / "deepfake-and-real-images"
    
    if not dataset_path.exists():
        print("❌ No datasets available for training")
        print("💡 Please ensure Kaggle API is configured and datasets are downloaded")
        return
    
    # Prepare dataset
    train_data, val_data, test_data, total_samples = trainer.prepare_dataset(dataset_path)
    
    if train_data is None:
        print("❌ Failed to prepare dataset")
        return
    
    # Create model
    model = trainer.create_advanced_model()
    
    # Create data generators
    train_gen, val_gen, steps_per_epoch, val_steps = trainer.create_data_generators(
        train_data, val_data, batch_size=16  # Smaller batch size for stability
    )
    
    # Train model
    print(f"\n🏋️  Training on {total_samples} images...")
    history = trainer.train_model(train_gen, val_gen, steps_per_epoch, val_steps, epochs=30)
    
    # Fine-tune model
    print(f"\n🔧 Fine-tuning model...")
    fine_tune_history = trainer.fine_tune_model(train_gen, val_gen, steps_per_epoch, val_steps, epochs=10)
    
    # Evaluate model
    if test_data:
        print(f"\n📊 Evaluating model...")
        results = trainer.evaluate_model(test_data)
        
        if results['accuracy'] >= 90:
            print("🎉 Target accuracy of 90%+ achieved!")
        else:
            print(f"📈 Current accuracy: {results['accuracy']:.2f}% - Continue training for better results")
    
    # Save model
    trainer.save_model("kaggle_enhanced_deepfake_detector.keras")
    
    print("\n🎉 Training completed successfully!")
    print("📁 Model saved and ready for use")


if __name__ == "__main__":
    main()
