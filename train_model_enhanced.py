#!/usr/bin/env python3
"""
Enhanced Model Training Script - Multiple Training Iterations
Trains the deepfake detection model multiple times with different configurations
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging

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
        """EfficientNet-based model"""
        inputs = layers.Input(shape=input_shape)
        
        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Backbone
        backbone = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)
        backbone.trainable = False
        
        x = backbone.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=predictions)
        return model
    
    def create_model_v2(self, input_shape=(224, 224, 3)):
        """ResNet-based model with attention"""
        inputs = layers.Input(shape=input_shape)
        
        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.15)(x)
        x = layers.RandomZoom(0.15)(x)
        x = layers.RandomContrast(0.1)(x)
        
        # Backbone
        backbone = ResNet50V2(weights='imagenet', include_top=False, input_tensor=x)
        for layer in backbone.layers[:-30]:
            layer.trainable = False
            
        x = backbone.output
        
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
        """Custom CNN with multiple scales"""
        inputs = layers.Input(shape=input_shape)
        
        # Data augmentation
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
        logger.info("Creating synthetic training data...")
        
        # Create directories
        os.makedirs('synthetic_data/train/real', exist_ok=True)
        os.makedirs('synthetic_data/train/fake', exist_ok=True)
        os.makedirs('synthetic_data/val/real', exist_ok=True)
        os.makedirs('synthetic_data/val/fake', exist_ok=True)
        
        # Generate synthetic images (random noise with patterns)
        for i in range(num_samples // 2):
            # Real images (more structured patterns)
            real_img = np.random.rand(224, 224, 3) * 0.5 + 0.3
            real_img = (real_img * 255).astype(np.uint8)
            
            # Fake images (more random patterns)
            fake_img = np.random.rand(224, 224, 3) * 0.8 + 0.1
            fake_img = (fake_img * 255).astype(np.uint8)
            
            # Save as numpy arrays (simplified for demo)
            if i < num_samples // 4:  # 25% for validation
                np.save(f'synthetic_data/val/real/real_{i}.npy', real_img)
                np.save(f'synthetic_data/val/fake/fake_{i}.npy', fake_img)
            else:
                np.save(f'synthetic_data/train/real/real_{i}.npy', real_img)
                np.save(f'synthetic_data/train/fake/fake_{i}.npy', fake_img)
        
        logger.info(f"Created {num_samples} synthetic samples")
    
    def create_data_generators(self, batch_size=32):
        """Create data generators for training"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # For demo purposes, create simple generators with synthetic data
        def generate_batch(batch_size, is_training=True):
            while True:
                batch_x = np.random.rand(batch_size, 224, 224, 3)
                batch_y = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)
                yield batch_x, batch_y
        
        train_gen = generate_batch(batch_size, True)
        val_gen = generate_batch(batch_size, False)
        
        return train_gen, val_gen
    
    def train_model_iteration(self, model_func, model_name, epochs=20):
        """Train a single model iteration"""
        logger.info(f"Starting training iteration: {model_name}")
        
        # Create model
        model = model_func()
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Create callbacks
        callbacks_list = [
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
            ),
            callbacks.ModelCheckpoint(
                f'{model_name}_best.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Get data generators
        train_gen, val_gen = self.create_data_generators()
        
        # Train model
        history = model.fit(
            train_gen,
            steps_per_epoch=50,  # Reduced for demo
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=20,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Get final accuracy
        final_accuracy = max(history.history['val_accuracy'])
        
        # Store results
        result = {
            'model_name': model_name,
            'final_accuracy': final_accuracy,
            'final_loss': min(history.history['val_loss']),
            'epochs_trained': len(history.history['accuracy']),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_results.append(result)
        
        # Check if this is the best model
        if final_accuracy > self.best_accuracy:
            self.best_accuracy = final_accuracy
            self.best_model = model_name
            model.save(f'best_model_{model_name}.keras')
        
        logger.info(f"Completed {model_name}: Accuracy = {final_accuracy:.4f}")
        return result
    
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
