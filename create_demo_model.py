#!/usr/bin/env python3
"""
Create a demo model for testing the deepfake detection system.
This creates a simple CNN model with the same architecture for demonstration.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os


def create_demo_model():
    """Create a demo model with the same architecture as the original."""
    print("🏗️  Creating demo model...")
    
    # Build the model architecture (same as in the original)
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1./255.0, name='rescaling'),  # Proper normalization
        
        # First conv block
        layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
        # Fourth conv block
        layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("✅ Model architecture created")
    print(f"   - Total parameters: {model.count_params():,}")
    
    return model


def create_dummy_training_data():
    """Create realistic dummy training data for the demo model."""
    print("📊 Creating realistic training data...")

    # Create more realistic training data with patterns
    batch_size = 200

    # Create "real" images with more natural patterns
    real_images = []
    for i in range(batch_size // 2):
        # Create images with more natural color distributions
        img = np.random.normal(128, 40, (128, 128, 3))
        # Add some structure to make it more realistic
        img[:, :, 0] += np.sin(np.linspace(0, 4*np.pi, 128)).reshape(-1, 1) * 20
        img[:, :, 1] += np.cos(np.linspace(0, 4*np.pi, 128)).reshape(1, -1) * 20
        img = np.clip(img, 0, 255).astype(np.uint8)
        real_images.append(img)

    # Create "fake" images with different patterns
    fake_images = []
    for i in range(batch_size // 2):
        # Create images with artificial patterns
        img = np.random.uniform(50, 200, (128, 128, 3))
        # Add artificial noise patterns
        noise = np.random.random((128, 128, 3)) * 100
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        fake_images.append(img)

    # Combine and create labels
    x_train = np.array(real_images + fake_images)
    y_train = np.array([0] * (batch_size // 2) + [1] * (batch_size // 2)).astype(np.float32)

    # Shuffle the data
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    # Create validation data with similar patterns
    val_size = 40
    x_val_real = []
    for i in range(val_size // 2):
        img = np.random.normal(120, 35, (128, 128, 3))
        img[:, :, 0] += np.sin(np.linspace(0, 3*np.pi, 128)).reshape(-1, 1) * 15
        img = np.clip(img, 0, 255).astype(np.uint8)
        x_val_real.append(img)

    x_val_fake = []
    for i in range(val_size // 2):
        img = np.random.uniform(60, 180, (128, 128, 3))
        noise = np.random.random((128, 128, 3)) * 80
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        x_val_fake.append(img)

    x_val = np.array(x_val_real + x_val_fake)
    y_val = np.array([0] * (val_size // 2) + [1] * (val_size // 2)).astype(np.float32)

    # Shuffle validation data
    val_indices = np.random.permutation(len(x_val))
    x_val = x_val[val_indices]
    y_val = y_val[val_indices]

    print("✅ Realistic training data created")
    print(f"   - Training samples: {len(x_train)} (Real: {batch_size//2}, Fake: {batch_size//2})")
    print(f"   - Validation samples: {len(x_val)} (Real: {val_size//2}, Fake: {val_size//2})")

    return (x_train, y_train), (x_val, y_val)


def train_demo_model(model, train_data, val_data, epochs=10):
    """Train the demo model with realistic data for better accuracy."""
    print(f"🏋️  Training enhanced demo model for {epochs} epochs...")

    x_train, y_train = train_data
    x_val, y_val = val_data

    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train the model with callbacks
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Enhanced training completed")
    print(f"   - Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"   - Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    return history


def save_demo_model(model, filename='deepfake_detector_model_demo.keras'):
    """Save the demo model."""
    print(f"💾 Saving demo model as {filename}...")
    
    model.save(filename)
    
    print("✅ Demo model saved successfully")
    print(f"   - File: {filename}")
    print(f"   - Size: {os.path.getsize(filename) / (1024*1024):.1f} MB")


def test_demo_model(model):
    """Test the demo model with a sample prediction."""
    print("🧪 Testing demo model...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8)
    
    # Make prediction
    prediction = model.predict(test_image, verbose=0)
    result = "Fake" if prediction[0][0] >= 0.5 else "Real"
    confidence = prediction[0][0] * 100 if prediction[0][0] >= 0.5 else (1 - prediction[0][0]) * 100
    
    print("✅ Model test successful")
    print(f"   - Prediction: {result}")
    print(f"   - Confidence: {confidence:.1f}%")


def main():
    """Main function to create and save the demo model."""
    print("🧠 Advanced Deepfake Detector - Demo Model Creator")
    print("=" * 60)
    
    # Check if demo model already exists
    demo_model_path = 'deepfake_detector_model_demo.keras'
    if os.path.exists(demo_model_path):
        print(f"⚠️  Demo model already exists: {demo_model_path}")
        overwrite = input("Do you want to overwrite it? (y/n): ").lower().strip()
        if overwrite not in ['y', 'yes']:
            print("❌ Cancelled by user")
            return
    
    try:
        # Create model
        model = create_demo_model()
        
        # Create dummy data
        train_data, val_data = create_dummy_training_data()
        
        # Train model (just a few epochs for demo)
        train_demo_model(model, train_data, val_data, epochs=3)
        
        # Test model
        test_demo_model(model)
        
        # Save model
        save_demo_model(model, demo_model_path)
        
        print("\n" + "=" * 60)
        print("🎉 Demo model created successfully!")
        print("=" * 60)
        print(f"📁 Model saved as: {demo_model_path}")
        print("🚀 You can now run the web application:")
        print("   python app.py")
        print("   (Make sure to update the model path in app.py if needed)")
        
    except Exception as e:
        print(f"\n❌ Error creating demo model: {e}")
        return


if __name__ == "__main__":
    main()
