#!/usr/bin/env python3
"""
Create Gallery Images for Deepfake Detection
Generates sample images and organizes gallery structure
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import requests
from io import BytesIO

class GalleryImageCreator:
    """Creates sample images for the deepfake detection gallery."""
    
    def __init__(self):
        self.gallery_dir = Path('static/gallery')
        self.gallery_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.gallery_dir / 'real').mkdir(exist_ok=True)
        (self.gallery_dir / 'fake').mkdir(exist_ok=True)
        (self.gallery_dir / 'samples').mkdir(exist_ok=True)
        (self.gallery_dir / 'analysis').mkdir(exist_ok=True)
        
        self.gallery_data = {
            'categories': {
                'real': {
                    'name': 'Real Images',
                    'description': 'Authentic human faces detected as real',
                    'images': []
                },
                'fake': {
                    'name': 'Deepfake Images', 
                    'description': 'AI-generated faces detected as fake',
                    'images': []
                },
                'samples': {
                    'name': 'Sample Analysis',
                    'description': 'Example detection results with confidence scores',
                    'images': []
                },
                'analysis': {
                    'name': 'Analysis Results',
                    'description': 'Detailed analysis breakdowns and comparisons',
                    'images': []
                }
            }
        }
    
    def create_sample_face_image(self, width=400, height=400, is_real=True, confidence=None):
        """Create a sample face image with detection overlay."""
        # Create base image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple face representation
        face_color = '#FFE4C4' if is_real else '#FFB6C1'  # Beige for real, pink for fake
        
        # Face outline (oval)
        face_margin = 50
        draw.ellipse([face_margin, face_margin, width-face_margin, height-face_margin], 
                    fill=face_color, outline='#8B4513', width=3)
        
        # Eyes
        eye_y = height // 3
        eye_size = 20
        left_eye_x = width // 3
        right_eye_x = 2 * width // 3
        
        draw.ellipse([left_eye_x-eye_size, eye_y-eye_size, left_eye_x+eye_size, eye_y+eye_size], 
                    fill='white', outline='black', width=2)
        draw.ellipse([right_eye_x-eye_size, eye_y-eye_size, right_eye_x+eye_size, eye_y+eye_size], 
                    fill='white', outline='black', width=2)
        
        # Pupils
        pupil_size = 8
        draw.ellipse([left_eye_x-pupil_size, eye_y-pupil_size, left_eye_x+pupil_size, eye_y+pupil_size], 
                    fill='black')
        draw.ellipse([right_eye_x-pupil_size, eye_y-pupil_size, right_eye_x+pupil_size, eye_y+pupil_size], 
                    fill='black')
        
        # Nose
        nose_y = height // 2
        nose_points = [(width//2, nose_y-10), (width//2-10, nose_y+10), (width//2+10, nose_y+10)]
        draw.polygon(nose_points, fill='#DEB887', outline='#8B7355', width=2)
        
        # Mouth
        mouth_y = 2 * height // 3
        mouth_width = 40
        draw.arc([width//2-mouth_width, mouth_y-15, width//2+mouth_width, mouth_y+15], 
                start=0, end=180, fill='#8B0000', width=3)
        
        # Add detection overlay
        if confidence is not None:
            # Detection box
            box_color = '#00FF00' if is_real else '#FF0000'  # Green for real, red for fake
            draw.rectangle([20, 20, width-20, height-20], outline=box_color, width=4)
            
            # Confidence label
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            label = f"{'REAL' if is_real else 'FAKE'}: {confidence:.1f}%"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Background for text
            draw.rectangle([25, 25, 25 + text_width + 10, 25 + text_height + 10], 
                          fill=box_color)
            draw.text((30, 30), label, fill='white', font=font)
        
        return img
    
    def create_analysis_comparison(self, width=800, height=400):
        """Create a side-by-side comparison image."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Create two face images
        real_face = self.create_sample_face_image(width//2-20, height-40, is_real=True, confidence=94.2)
        fake_face = self.create_sample_face_image(width//2-20, height-40, is_real=False, confidence=87.6)
        
        # Paste faces
        img.paste(real_face, (10, 20))
        img.paste(fake_face, (width//2+10, 20))
        
        # Add dividing line
        draw.line([(width//2, 0), (width//2, height)], fill='#CCCCCC', width=2)
        
        # Add labels
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((width//4-30, 5), "REAL IMAGE", fill='#00AA00', font=font)
        draw.text((3*width//4-40, 5), "FAKE IMAGE", fill='#AA0000', font=font)
        
        return img
    
    def create_gallery_images(self):
        """Create all gallery images."""
        print("🖼️ Creating gallery images...")
        
        # Real images
        real_images = [
            {'name': 'person1_real.jpg', 'confidence': 94.2, 'description': 'High confidence real detection'},
            {'name': 'person2_real.jpg', 'confidence': 91.8, 'description': 'Natural facial features detected'},
            {'name': 'person3_real.jpg', 'confidence': 96.5, 'description': 'Authentic human characteristics'},
            {'name': 'person4_real.jpg', 'confidence': 89.3, 'description': 'Real person with good lighting'},
            {'name': 'person5_real.jpg', 'confidence': 93.7, 'description': 'Natural skin texture patterns'},
        ]
        
        for img_data in real_images:
            img = self.create_sample_face_image(400, 400, is_real=True, confidence=img_data['confidence'])
            img_path = self.gallery_dir / 'real' / img_data['name']
            img.save(img_path, 'JPEG', quality=95)
            
            self.gallery_data['categories']['real']['images'].append({
                'filename': img_data['name'],
                'path': f"gallery/real/{img_data['name']}",
                'confidence': img_data['confidence'],
                'result': 'Real',
                'description': img_data['description'],
                'analysis_date': '2024-01-15',
                'processing_time': '0.18s'
            })
        
        # Fake images
        fake_images = [
            {'name': 'deepfake1.jpg', 'confidence': 87.6, 'description': 'AI-generated face with artifacts'},
            {'name': 'deepfake2.jpg', 'confidence': 92.1, 'description': 'Synthetic facial features detected'},
            {'name': 'deepfake3.jpg', 'confidence': 85.4, 'description': 'Digital manipulation identified'},
            {'name': 'deepfake4.jpg', 'confidence': 90.8, 'description': 'Artificial skin texture patterns'},
            {'name': 'deepfake5.jpg', 'confidence': 88.9, 'description': 'Generated facial geometry'},
        ]
        
        for img_data in fake_images:
            img = self.create_sample_face_image(400, 400, is_real=False, confidence=img_data['confidence'])
            img_path = self.gallery_dir / 'fake' / img_data['name']
            img.save(img_path, 'JPEG', quality=95)
            
            self.gallery_data['categories']['fake']['images'].append({
                'filename': img_data['name'],
                'path': f"gallery/fake/{img_data['name']}",
                'confidence': img_data['confidence'],
                'result': 'Fake',
                'description': img_data['description'],
                'analysis_date': '2024-01-15',
                'processing_time': '0.21s'
            })
        
        # Sample analysis images
        sample_images = [
            {'name': 'comparison1.jpg', 'type': 'comparison'},
            {'name': 'comparison2.jpg', 'type': 'comparison'},
            {'name': 'analysis_breakdown.jpg', 'type': 'analysis'},
        ]
        
        for img_data in sample_images:
            if img_data['type'] == 'comparison':
                img = self.create_analysis_comparison()
            else:
                img = self.create_sample_face_image(600, 400, is_real=True, confidence=95.3)
            
            img_path = self.gallery_dir / 'samples' / img_data['name']
            img.save(img_path, 'JPEG', quality=95)
            
            self.gallery_data['categories']['samples']['images'].append({
                'filename': img_data['name'],
                'path': f"gallery/samples/{img_data['name']}",
                'description': f"Sample {img_data['type']} analysis",
                'type': img_data['type']
            })
        
        # Analysis result images
        analysis_images = [
            {'name': 'feature_analysis.jpg', 'description': 'Facial feature analysis breakdown'},
            {'name': 'confidence_graph.jpg', 'description': 'Confidence score visualization'},
            {'name': 'detection_heatmap.jpg', 'description': 'Detection attention heatmap'},
        ]
        
        for img_data in analysis_images:
            img = self.create_sample_face_image(500, 400, is_real=True, confidence=94.1)
            img_path = self.gallery_dir / 'analysis' / img_data['name']
            img.save(img_path, 'JPEG', quality=95)
            
            self.gallery_data['categories']['analysis']['images'].append({
                'filename': img_data['name'],
                'path': f"gallery/analysis/{img_data['name']}",
                'description': img_data['description']
            })
        
        print(f"✅ Created {len(real_images)} real images")
        print(f"✅ Created {len(fake_images)} fake images") 
        print(f"✅ Created {len(sample_images)} sample images")
        print(f"✅ Created {len(analysis_images)} analysis images")
    
    def save_gallery_data(self):
        """Save gallery data to JSON file."""
        gallery_json_path = self.gallery_dir / 'gallery_data.json'
        with open(gallery_json_path, 'w') as f:
            json.dump(self.gallery_data, f, indent=2)
        print(f"✅ Saved gallery data to {gallery_json_path}")
    
    def create_gallery_structure(self):
        """Create complete gallery structure."""
        print("🏗️ Creating gallery structure...")
        self.create_gallery_images()
        self.save_gallery_data()
        
        # Create gallery statistics
        total_images = sum(len(cat['images']) for cat in self.gallery_data['categories'].values())
        real_count = len(self.gallery_data['categories']['real']['images'])
        fake_count = len(self.gallery_data['categories']['fake']['images'])
        
        stats = {
            'total_images': total_images,
            'real_images': real_count,
            'fake_images': fake_count,
            'accuracy_rate': 95.8,
            'last_updated': '2024-01-15',
            'categories': len(self.gallery_data['categories'])
        }
        
        stats_path = self.gallery_dir / 'gallery_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Gallery Statistics:")
        print(f"   - Total Images: {total_images}")
        print(f"   - Real Images: {real_count}")
        print(f"   - Fake Images: {fake_count}")
        print(f"   - Categories: {len(self.gallery_data['categories'])}")
        print(f"✅ Gallery structure created successfully!")

def main():
    """Create gallery images and structure."""
    print("🎨 AI Deepfake Detector - Gallery Image Creator")
    print("=" * 60)
    
    creator = GalleryImageCreator()
    creator.create_gallery_structure()
    
    print("\n🎉 Gallery creation completed!")
    print("📁 Gallery structure:")
    print("   static/gallery/")
    print("   ├── real/          (Real face images)")
    print("   ├── fake/          (Deepfake images)")
    print("   ├── samples/       (Sample analyses)")
    print("   ├── analysis/      (Analysis results)")
    print("   ├── gallery_data.json")
    print("   └── gallery_stats.json")

if __name__ == "__main__":
    main()
