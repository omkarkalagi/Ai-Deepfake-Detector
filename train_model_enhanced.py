#!/usr/bin/env python3
"""
Enhanced Model Training Script - Multiple Training Iterations
Trains the deepfake detection model multiple times with different configurations
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTrainingSession:
    def __init__(self):
        self.training_results = []
        self.best_model = None
        self.best_accuracy = 0.0
        self.current_session = None
        self.checkpoint_dir = "checkpoints"
        self.session_file = "training_session.json"
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_session_state(self, session_data):
        """Save current training session state"""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            logger.info(f"Session state saved to {self.session_file}")
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
    
    def load_session_state(self):
        """Load previous training session state"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                logger.info(f"Session state loaded from {self.session_file}")
                return session_data
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
        return None
    
    def can_resume_training(self):
        """Check if there's a training session that can be resumed"""
        session_data = self.load_session_state()
        if session_data and session_data.get('status') == 'interrupted':
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{session_data['model_name']}_checkpoint.h5")
            return os.path.exists(checkpoint_path), session_data
        return False, None
    
    def resume_training(self, session_data, remaining_epochs):
        """Resume training from checkpoint"""
        try:
            model_name = session_data['model_name']
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_checkpoint.h5")
            
            # Load model from checkpoint
            model = tf.keras.models.load_model(checkpoint_path)
            logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
            
            # Load training data (recreate synthetic data)
            self.create_synthetic_data(1000)
            
            # Continue training
            completed_epochs = session_data.get('completed_epochs', 0)
            total_epochs = session_data.get('epochs', 10)
            
            # Update session status
            session_data['status'] = 'training'
            session_data['resumed_at'] = datetime.now().isoformat()
            self.save_session_state(session_data)
            
            # Train for remaining epochs
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=remaining_epochs,
                batch_size=session_data.get('batch_size', 32),
                verbose=1,
                callbacks=[
                    callbacks.ModelCheckpoint(
                        checkpoint_path,
                        save_best_only=True,
                        monitor='val_accuracy'
                    ),
                    callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Update session with completion
            final_accuracy = max(history.history['val_accuracy'])
            session_data['completed_epochs'] = total_epochs
            session_data['final_accuracy'] = float(final_accuracy)
            session_data['status'] = 'completed'
            session_data['completed_at'] = datetime.now().isoformat()
            self.save_session_state(session_data)
            
            # Save final model
            final_model_path = f"{model_name}_final.h5"
            model.save(final_model_path)
            
            result = {
                'model_name': model_name,
                'final_accuracy': final_accuracy,
                'completed_epochs': total_epochs,
                'model_path': final_model_path,
                'resumed': True
            }
            
            logger.info(f"Training resumed and completed: {final_accuracy:.4f} accuracy")
            return result
            
        except Exception as e:
            # Mark session as failed
            session_data['status'] = 'failed'
            session_data['error'] = str(e)
            self.save_session_state(session_data)
            logger.error(f"Failed to resume training: {e}")
            raise e
        
    def create_model_v1(self, input_shape=(224, 224, 3)):
        """Simple CNN model to avoid EfficientNet weight conflicts"""
        # Clear session first
        tf.keras.backend.clear_session()
        
        inputs = layers.Input(shape=input_shape)

        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)

        # Lightweight CNN architecture optimized for cloud deployment
        # Reduce model complexity for Railway constraints
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        
        # Simplified dense layers for Railway optimization
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output with proper initialization
        predictions = layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='glorot_uniform'
        )(x)

        model = models.Model(inputs=inputs, outputs=predictions)
        return model
    
    def create_model_v2(self, input_shape=(224, 224, 3)):
        """ResNet-based model with attention (3-channel enforced)"""
        inputs = layers.Input(shape=input_shape)

        # Enhanced data augmentation
        aug = layers.RandomFlip("horizontal")(inputs)
        aug = layers.RandomRotation(0.2)(aug)
        aug = layers.RandomZoom(0.2)(aug)
        aug = layers.RandomBrightness(0.2)(aug)
        aug = layers.RandomContrast(0.2)(aug)
        aug = layers.RandomRotation(0.15)(aug)
        aug = layers.RandomZoom(0.15)(aug)
        aug = layers.RandomContrast(0.1)(aug)

        # Backbone with explicit input_shape and functional call
        # Use randomly initialized weights to avoid channel mismatch from pre-trained weights
        backbone = ResNet50V2(weights=None, include_top=False, input_shape=input_shape)
        for layer in backbone.layers[:-30]:
            layer.trainable = False

        x = backbone(aug)

        # Attention mechanism
        attention = layers.Dense(x.shape[-1], activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=predictions)
        return model
    
    def create_model_v3(self, input_shape=(224, 224, 3)):
        """Custom CNN with multiple scales (3-channel enforced)"""
        inputs = layers.Input(shape=input_shape)

        # Data augmentation on inputs
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomBrightness(0.1)(x)

        # Multi-scale feature extraction
        # Scale 1: Fine details
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.MaxPooling2D((2, 2))(conv1)

        # Scale 2: Medium features
        conv2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.MaxPooling2D((2, 2))(conv2)

        # Scale 3: Large features
        conv3 = layers.Conv2D(128, (7, 7), activation='relu', padding='same')(conv2)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.MaxPooling2D((2, 2))(conv3)

        # Additional layers
        conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        conv4 = layers.BatchNormalization()(conv4)
        conv4 = layers.MaxPooling2D((2, 2))(conv4)

        conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        conv5 = layers.BatchNormalization()(conv5)
        conv5 = layers.GlobalAveragePooling2D()(conv5)

        # Dense layers
        dense1 = layers.Dense(1024, activation='relu')(conv5)
        dense1 = layers.Dropout(0.5)(dense1)
        dense2 = layers.Dense(512, activation='relu')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        predictions = layers.Dense(1, activation='sigmoid')(dense2)

        model = models.Model(inputs=inputs, outputs=predictions)
        return model
    
    def create_synthetic_data(self, num_samples=1000):
        """Create synthetic training data for demonstration"""
        try:
            logger.info("Creating synthetic training data...")
            
            # Clean up existing synthetic data
            if os.path.exists('synthetic_data'):
                shutil.rmtree('synthetic_data')
            
            # Create fresh directories
            os.makedirs('synthetic_data/train/real', exist_ok=True)
            os.makedirs('synthetic_data/train/fake', exist_ok=True)
            os.makedirs('synthetic_data/val/real', exist_ok=True)
            os.makedirs('synthetic_data/val/fake', exist_ok=True)
            
            logger.info("Created directory structure for synthetic data")
            
            # Generate synthetic images with meaningful patterns
            for i in range(num_samples // 2):
                # Real images - natural face-like patterns
                real_img = np.zeros((224, 224, 3))
                
                # Create skin-tone base
                skin_tone = np.random.uniform(0.5, 0.8)
                real_img[:, :] = [skin_tone * 255, (skin_tone - 0.1) * 255, (skin_tone - 0.15) * 255]
                
                # Add facial features (simplified)
                center_y, center_x = 112, 112
                # Eyes
                cv2.circle(real_img, (center_x - 40, center_y - 20), 15, (50, 50, 150), -1)
                cv2.circle(real_img, (center_x + 40, center_y - 20), 15, (50, 50, 150), -1)
                # Mouth
                cv2.ellipse(real_img, (center_x, center_y + 30), (40, 20), 0, 0, 180, (150, 50, 50), -1)
                
                # Add natural variations
                real_img += np.random.normal(0, 10, (224, 224, 3))
                real_img = np.clip(real_img, 0, 255).astype(np.uint8)
                
                # Fake images - unnatural patterns and artifacts
                fake_img = np.zeros((224, 224, 3))
                
                # Create base with unnatural color variations
                fake_base = np.random.uniform(0.4, 0.9)
                fake_img[:, :] = [fake_base * 255, (fake_base + 0.2) * 255, (fake_base - 0.2) * 255]
                
                # Add artifacts and inconsistencies
                for _ in range(5):
                    x1, y1 = np.random.randint(0, 224, 2)
                    x2, y2 = np.random.randint(0, 224, 2)
                    color = tuple(map(int, np.random.randint(0, 255, 3)))
                    cv2.line(fake_img, (x1, y1), (x2, y2), color, 2)
                
                # Add digital noise patterns
                fake_img += np.random.normal(0, 20, (224, 224, 3))
                fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
                
                # Save images
                if i < num_samples // 4:  # 25% for validation
                    cv2.imwrite(f'synthetic_data/val/real/real_{i}.jpg', real_img)
                    cv2.imwrite(f'synthetic_data/val/fake/fake_{i}.jpg', fake_img)
                else:
                    cv2.imwrite(f'synthetic_data/train/real/real_{i}.jpg', real_img)
                    cv2.imwrite(f'synthetic_data/train/fake/fake_{i}.jpg', fake_img)
        
            logger.info(f"Created {num_samples} synthetic samples")
        except Exception as e:
            logger.error(f"Error creating synthetic data: {str(e)}")
            raise
    
    def create_data_generators(self, batch_size=32):
        """Create data generators for training"""
        # Enhanced data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=20,
            fill_mode='reflect',
            validation_split=0.0  # We have a separate validation set
        )
        
        # Minimal processing for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators from image directories
        train_generator = train_datagen.flow_from_directory(
            'synthetic_data/train',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            'synthetic_data/val',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train_model_iteration(self, model, model_name, epochs=20):
        """Train a single model iteration"""
        logger.info(f"Starting training iteration: {model_name}")
        
        try:
            # Clear any existing session to avoid weight conflicts
            tf.keras.backend.clear_session()
            
            # Compile the model before training
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Get data generators
            train_gen, val_gen = self.create_data_generators(batch_size=32)
            
            # Custom callback to update session state during training
            class SessionUpdateCallback(callbacks.Callback):
                def __init__(self, trainer, model_name, total_epochs):
                    super().__init__()
                    self.trainer = trainer
                    self.model_name = model_name
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    # Update session state with current progress
                    session_data = {
                        'model_name': self.model_name,
                        'epochs': self.total_epochs,
                        'completed_epochs': epoch + 1,
                        'status': 'training',
                        'current_accuracy': float(logs.get('accuracy', 0)),
                        'current_val_accuracy': float(logs.get('val_accuracy', 0)),
                        'current_loss': float(logs.get('loss', 0)),
                        'current_val_loss': float(logs.get('val_loss', 0)),
                        'progress': ((epoch + 1) / self.total_epochs) * 100,
                        'last_update': datetime.now().isoformat()
                    }
                    self.trainer.save_session_state(session_data)
            
            # Create callbacks
            callbacks_list = [
                SessionUpdateCallback(self, model_name, epochs),
                callbacks.ModelCheckpoint(
                    f'{model_name}_best.keras',
                    monitor='val_accuracy',
                    save_best_only=True
                ),
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Get final metrics
            final_metrics = {
                'accuracy': float(history.history['accuracy'][-1]),
                'val_accuracy': float(history.history['val_accuracy'][-1]),
                'loss': float(history.history['loss'][-1]),
                'val_loss': float(history.history['val_loss'][-1])
            }
            
            logger.info(f"Training completed. Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
            
            # Store results
            result = {
                'model_name': model_name,
                'final_accuracy': final_metrics['val_accuracy'],
                'final_loss': final_metrics['val_loss'],
                'epochs_trained': len(history.history['accuracy']),
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_results.append(result)
            
            # Check if this is the best model
            if final_metrics['val_accuracy'] > self.best_accuracy:
                self.best_accuracy = final_metrics['val_accuracy']
                self.best_model = model_name
                model.save(f'best_model_{model_name}.keras')
            
            logger.info(f"Completed {model_name}: Accuracy = {final_metrics['val_accuracy']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {str(e)}")
            raise
    
    def run_multiple_training_sessions(self):
        """Run multiple training sessions with different models"""
        logger.info("Starting multiple training sessions...")
        
        # Training configurations
        training_configs = [
            (self.create_model_v1, "EfficientNet_v1", 15),
            (self.create_model_v2, "ResNet_Attention_v2", 20),
            (self.create_model_v3, "CustomCNN_v3", 25),
            (self.create_model_v1, "EfficientNet_v1_extended", 30),
            (self.create_model_v2, "ResNet_Attention_v2_extended", 35),
        ]
        
        # Run each training configuration
        for model_func, model_name, epochs in training_configs:
            try:
                result = self.train_model_iteration(model_func, model_name, epochs)
                logger.info(f"Training session completed: {result}")
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {e}")
        
        # Save results
        self.save_training_results()
        self.display_results()
    
    def save_training_results(self):
        """Save training results to file"""
        results_summary = {
            'training_sessions': self.training_results,
            'best_model': self.best_model,
            'best_accuracy': self.best_accuracy,
            'total_sessions': len(self.training_results),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('training_results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info("Training results saved to training_results_summary.json")
    
    def display_results(self):
        """Display training results summary"""
        print("\n" + "="*60)
        print("MULTIPLE TRAINING SESSIONS SUMMARY")
        print("="*60)
        
        for i, result in enumerate(self.training_results, 1):
            print(f"\nSession {i}: {result['model_name']}")
            print(f"  Accuracy: {result['final_accuracy']:.4f}")
            print(f"  Loss: {result['final_loss']:.4f}")
            print(f"  Epochs: {result['epochs_trained']}")
        
        print(f"\nBEST MODEL: {self.best_model}")
        print(f"BEST ACCURACY: {self.best_accuracy:.4f}")
        print("="*60)

def main():
    """Main training function"""
    print("Starting Enhanced Multi-Training Session...")
    
    # Initialize training session
    trainer = MultiTrainingSession()
    
    # Create synthetic data for demonstration
    trainer.create_synthetic_data(2000)
    
    # Run multiple training sessions
    trainer.run_multiple_training_sessions()
    
    print("Multi-training session completed!")

if __name__ == "__main__":
    main()
