#!/usr/bin/env python3
"""
Test script to verify model creation works without shape conflicts
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def test_model_creation():
    """Test creating a fresh EfficientNet model"""
    try:
        print("Testing model creation...")
        
        # Clear any existing session
        tf.keras.backend.clear_session()
        
        input_shape = (224, 224, 3)
        inputs = layers.Input(shape=input_shape)

        # Data augmentation
        aug = layers.RandomFlip("horizontal")(inputs)
        aug = layers.RandomRotation(0.1)(aug)
        aug = layers.RandomZoom(0.1)(aug)

        # Create backbone without pre-trained weights
        backbone = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
        
        backbone.trainable = True
        
        # Feature extraction
        x = backbone(aug)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        predictions = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model created successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Total parameters: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_creation()
    if success:
        print("Model creation test passed!")
    else:
        print("Model creation test failed!")
