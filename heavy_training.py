"""
Heavy Training Script for Maximum Deepfake Detection Accuracy
Optimized specifically for detecting thispersondoesnotexist.com images
"""

import os
import numpy as np
import cv2
import json
from datetime import datetime
import logging
from pathlib import Path
import requests
import time

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeavyTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.training_dir = Path("heavy_training_data")
        self.training_dir.mkdir(exist_ok=True)
        
    def download_ai_faces(self, num_samples=500):
        """Download real AI faces from thispersondoesnotexist.com"""
        logger.info(f"Downloading {num_samples} AI faces from thispersondoesnotexist.com...")
        
        fake_train = self.training_dir / "train" / "fake"
        fake_val = self.training_dir / "val" / "fake"
        fake_train.mkdir(parents=True, exist_ok=True)
        fake_val.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        for i in range(num_samples):
            try:
                url = "https://thispersondoesnotexist.com/image"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
                response = requests.get(url, headers=headers, timeout=20)
                if response.status_code == 200:
                    # 80% train, 20% validation
                    if i % 5 == 0:
                        save_path = fake_val / f"ai_{i:04d}.jpg"
                    else:
                        save_path = fake_train / f"ai_{i:04d}.jpg"
                    
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    
                    downloaded += 1
                    
                    if downloaded % 25 == 0:
                        logger.info(f"Downloaded {downloaded} AI faces...")
                        
                    time.sleep(2)  # Respectful delay
                    
            except Exception as e:
                logger.warning(f"Failed to download sample {i}: {e}")
                continue
                
        logger.info(f"Downloaded {downloaded} AI faces")
        return downloaded
    
    def create_real_faces(self, num_samples=500):
        """Create diverse realistic human faces"""
        logger.info(f"Creating {num_samples} realistic human faces...")
        
        real_train = self.training_dir / "train" / "real"
        real_val = self.training_dir / "val" / "real"
        real_train.mkdir(parents=True, exist_ok=True)
        real_val.mkdir(parents=True, exist_ok=True)
        
        created = 0
        for i in range(num_samples):
            try:
                img = self.create_human_face()
                
                # 80% train, 20% validation
                if i % 5 == 0:
                    save_path = real_val / f"human_{i:04d}.jpg"
                else:
                    save_path = real_train / f"human_{i:04d}.jpg"
                
                cv2.imwrite(str(save_path), img)
                created += 1
                
                if created % 50 == 0:
                    logger.info(f"Created {created} realistic faces...")
                    
            except Exception as e:
                continue
                
        logger.info(f"Created {created} realistic faces")
        return created
    
    def create_human_face(self):
        """Create realistic human face with natural variations"""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Diverse skin tones
        ethnicity = np.random.choice(['caucasian', 'african', 'asian', 'hispanic', 'mixed'])
        if ethnicity == 'caucasian':
            base_r, base_g, base_b = np.random.randint(200, 240), np.random.randint(180, 220), np.random.randint(160, 200)
        elif ethnicity == 'african':
            base_r, base_g, base_b = np.random.randint(80, 140), np.random.randint(60, 120), np.random.randint(40, 100)
        elif ethnicity == 'asian':
            base_r, base_g, base_b = np.random.randint(180, 220), np.random.randint(160, 200), np.random.randint(140, 180)
        elif ethnicity == 'hispanic':
            base_r, base_g, base_b = np.random.randint(150, 200), np.random.randint(120, 170), np.random.randint(100, 150)
        else:  # mixed
            base_r, base_g, base_b = np.random.randint(140, 210), np.random.randint(110, 190), np.random.randint(90, 170)
        
        # Highly irregular face shape
        center_x = 128 + np.random.randint(-20, 21)
        center_y = 128 + np.random.randint(-20, 21)
        width = 80 + np.random.randint(-25, 26)
        height = 100 + np.random.randint(-25, 26)
        angle = np.random.randint(-45, 46)
        
        cv2.ellipse(img, (center_x, center_y), (width, height), angle, 0, 360, (base_b, base_g, base_r), -1)
        
        # Asymmetric eyes
        eye1_x, eye1_y = 90 + np.random.randint(-15, 16), 110 + np.random.randint(-20, 21)
        eye2_x, eye2_y = 166 + np.random.randint(-15, 16), 110 + np.random.randint(-20, 21)
        eye1_w, eye1_h = 15 + np.random.randint(-5, 6), 10 + np.random.randint(-4, 5)
        eye2_w, eye2_h = 15 + np.random.randint(-5, 6), 10 + np.random.randint(-4, 5)
        
        cv2.ellipse(img, (eye1_x, eye1_y), (eye1_w, eye1_h), np.random.randint(-20, 21), 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (eye2_x, eye2_y), (eye2_w, eye2_h), np.random.randint(-20, 21), 0, 360, (255, 255, 255), -1)
        
        # Irregular pupils
        cv2.circle(img, (eye1_x, eye1_y), 7 + np.random.randint(-3, 4), (50, 50, 50), -1)
        cv2.circle(img, (eye2_x, eye2_y), 7 + np.random.randint(-3, 4), (50, 50, 50), -1)
        
        # Asymmetric nose
        nose_x = center_x + np.random.randint(-10, 11)
        nose_y = 140 + np.random.randint(-15, 16)
        nose_points = np.array([
            [nose_x, nose_y], 
            [nose_x - 6 + np.random.randint(-4, 5), nose_y + 20 + np.random.randint(-8, 9)], 
            [nose_x + 6 + np.random.randint(-4, 5), nose_y + 20 + np.random.randint(-8, 9)]
        ], np.int32)
        cv2.fillPoly(img, [nose_points], (base_b-40, base_g-40, base_r-40))
        
        # Asymmetric mouth
        mouth_x = center_x + np.random.randint(-15, 16)
        mouth_y = 180 + np.random.randint(-15, 16)
        mouth_w = 25 + np.random.randint(-8, 9)
        mouth_h = 8 + np.random.randint(-3, 4)
        cv2.ellipse(img, (mouth_x, mouth_y), (mouth_w, mouth_h), np.random.randint(-30, 31), 0, 360, (120, 80, 80), -1)
        
        # Natural imperfections
        for _ in range(300):
            x, y = np.random.randint(40, 216, 2)
            if 40 < x < 216 and 40 < y < 216:
                intensity = np.random.randint(1, 5)
                color_var = np.random.randint(-50, 51)
                cv2.circle(img, (x, y), intensity, 
                          (max(0, min(255, base_b + color_var)),
                           max(0, min(255, base_g + color_var)),
                           max(0, min(255, base_r + color_var))), -1)
        
        # Complex lighting
        light_center_x = np.random.randint(64, 192)
        light_center_y = np.random.randint(64, 192)
        
        for y in range(256):
            for x in range(256):
                if not np.array_equal(img[y, x], [0, 0, 0]):
                    distance = np.sqrt((x - light_center_x)**2 + (y - light_center_y)**2)
                    light_factor = 0.4 + 0.8 * np.exp(-distance / 100)
                    img[y, x] = np.clip(img[y, x] * light_factor, 0, 255)
        
        # Heavy noise and compression
        noise = np.random.normal(0, 20, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Multiple compression passes
        for _ in range(np.random.randint(1, 4)):
            quality = np.random.randint(60, 90)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encimg, 1)
        
        # Resize to training size
        img = cv2.resize(img, (224, 224))
        
        return img
    
    def create_heavy_model(self):
        """Create heavy-duty model for maximum accuracy"""
        inputs = layers.Input(shape=(224, 224, 3))
        
        # Minimal augmentation to preserve real features
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.02)(x)
        
        # Block 1 - Low-level feature detection
        x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        # Block 2 - Texture analysis
        x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.15)(x)
        
        # Block 3 - Pattern recognition
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Block 4 - High-level features
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 5 - Deep analysis
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Classification layers
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def train_heavy_model(self, epochs=100):
        """Train heavy model for maximum accuracy"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available!")
            return None
            
        logger.info("Starting heavy training for maximum accuracy...")
        
        # Create datasets
        ai_count = self.download_ai_faces(400)
        if ai_count < 50:
            logger.warning("Failed to download AI faces, creating synthetic ones...")
            ai_count = self.create_synthetic_ai_faces(400)
        
        real_count = self.create_real_faces(400)
        
        if ai_count < 50 or real_count < 50:
            logger.error("Insufficient training data!")
            return None
        
        # Create heavy model
        self.model = self.create_heavy_model()
        
        # Compile with conservative settings
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Conservative augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=3,
            width_shift_range=0.02,
            height_shift_range=0.02,
            horizontal_flip=True,
            brightness_range=[0.95, 1.05]
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            str(self.training_dir / "train"),
            target_size=(224, 224),
            batch_size=4,  # Small batch for stability
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            str(self.training_dir / "val"),
            target_size=(224, 224),
            batch_size=4,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=False
        )
        
        # Heavy training callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max'),
            callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=10, min_lr=0.000001, mode='max'),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.000001)
        ]
        
        # Train model
        logger.info(f"Training heavy model for {epochs} epochs...")
        start_time = time.time()
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Save model weights
        self.model.save_weights('heavy_deepfake_detector.weights.h5')
        
        # Evaluate
        val_loss, val_acc, val_prec, val_rec = self.model.evaluate(val_generator, verbose=0)
        f1_score = 2 * (val_prec * val_rec) / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0
        
        # Save final model
        final_weights_path = f"heavy_deepfake_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.weights.h5"
        self.model.save_weights(final_weights_path)
        
        # Results
        results = {
            'model_path': final_weights_path,
            'training_date': datetime.now().isoformat(),
            'final_accuracy': float(val_acc),
            'final_precision': float(val_prec),
            'final_recall': float(val_rec),
            'f1_score': float(f1_score),
            'epochs_trained': len(self.history.history['accuracy']),
            'training_time_seconds': training_time,
            'best_val_accuracy': float(max(self.history.history['val_accuracy'])),
            'training_samples': {
                'ai_faces': ai_count,
                'real_faces': real_count,
                'total': ai_count + real_count
            }
        }
        
        # Save results
        with open('heavy_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Heavy training completed in {training_time:.1f} seconds!")
        logger.info(f"Final validation accuracy: {val_acc:.4f}")
        logger.info(f"Final precision: {val_prec:.4f}")
        logger.info(f"Final recall: {val_rec:.4f}")
        logger.info(f"F1 Score: {f1_score:.4f}")
        
        return results
    
    def create_synthetic_ai_faces(self, num_samples):
        """Fallback: create synthetic AI-style faces"""
        logger.info(f"Creating {num_samples} synthetic AI-style faces...")
        
        fake_train = self.training_dir / "train" / "fake"
        fake_val = self.training_dir / "val" / "fake"
        fake_train.mkdir(parents=True, exist_ok=True)
        fake_val.mkdir(parents=True, exist_ok=True)
        
        created = 0
        for i in range(num_samples):
            try:
                img = self.create_ai_style_face()
                
                if i % 5 == 0:
                    save_path = fake_val / f"synthetic_ai_{i:04d}.jpg"
                else:
                    save_path = fake_train / f"synthetic_ai_{i:04d}.jpg"
                
                cv2.imwrite(str(save_path), img)
                created += 1
                
            except Exception as e:
                continue
                
        logger.info(f"Created {created} synthetic AI-style faces")
        return created
    
    def create_ai_style_face(self):
        """Create AI-style face with typical artifacts"""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Perfect gradients (AI characteristic)
        base_color = np.random.randint(180, 220, 3)
        for y in range(224):
            for x in range(224):
                gradient_factor = 0.8 + 0.4 * np.sin(x/30) * np.cos(y/30)
                img[y, x] = np.clip(base_color * gradient_factor, 0, 255)
        
        # Perfect symmetric face
        center = (112, 112)
        cv2.ellipse(img, center, (75, 95), 0, 0, 360, (210, 190, 170), -1)
        
        # Perfect symmetric eyes
        cv2.ellipse(img, (87, 95), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (137, 95), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (87, 95), 6, (50, 50, 50), -1)
        cv2.circle(img, (137, 95), 6, (50, 50, 50), -1)
        
        # Perfect nose
        nose_points = np.array([[112, 105], [108, 120], [116, 120]], np.int32)
        cv2.fillPoly(img, [nose_points], (190, 170, 150))
        
        # Perfect mouth
        cv2.ellipse(img, (112, 135), (18, 6), 0, 0, 360, (140, 100, 100), -1)
        
        # AI artifacts
        # 1. Checkerboard pattern
        for y in range(0, 224, 16):
            for x in range(0, 224, 16):
                if (x//16 + y//16) % 2 == 0:
                    img[y:y+4, x:x+4] = np.clip(img[y:y+4, x:x+4] + 15, 0, 255)
        
        # 2. Gaussian blur
        img = cv2.GaussianBlur(img, (3, 3), 1.0)
        
        # 3. Color quantization
        img = (img // 16) * 16
        
        return img

def main():
    """Main training function"""
    logger.info("Starting Heavy Training for Maximum Accuracy")
    
    trainer = HeavyTrainer()
    results = trainer.train_heavy_model(epochs=100)
    
    if results:
        logger.info("Heavy training completed successfully!")
        logger.info(f"Model saved as: {results['model_path']}")
        logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"F1 Score: {results['f1_score']:.4f}")
    else:
        logger.error("Heavy training failed!")

if __name__ == "__main__":
    main()
