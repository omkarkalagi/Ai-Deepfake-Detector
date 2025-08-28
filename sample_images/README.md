# Sample Images for Deepfake Detection Testing

This folder contains 10 sample images for testing the AI Deepfake Detector:

## Real Images (5)
- `real_person_1.jpg` - Authentic photograph of a person
- `real_person_2.jpg` - Natural lighting, unedited photo
- `real_person_3.jpg` - High-quality authentic portrait
- `real_person_4.jpg` - Candid real photograph
- `real_person_5.jpg` - Professional but authentic photo

## Fake/Deepfake Images (5)
- `fake_person_1.jpg` - AI-generated face (StyleGAN/ThisPersonDoesNotExist)
- `fake_person_2.jpg` - Deepfake manipulation
- `fake_person_3.jpg` - Face swap technology result
- `fake_person_4.jpg` - AI-generated synthetic face
- `fake_person_5.jpg` - Digital manipulation/editing

## Usage Instructions
1. Upload these images to the AI Deepfake Detector
2. Compare the detection results
3. Real images should show high confidence for "Real"
4. Fake images should show high confidence for "Fake"
5. Use these for testing accuracy and model performance

## Expected Results
- Real images: 85-95% confidence as "Real"
- Fake images: 80-90% confidence as "Fake"
- Processing time: 2-5 seconds per image

Note: These are sample images for testing purposes. Actual detection accuracy may vary based on image quality, lighting, and other factors.
