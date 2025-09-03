#!/usr/bin/env python3
"""
Cloud Upload Script for Project Documents
Uploads PPT and PDF to cloud storage for easy access
"""

import os
import requests
import json
from datetime import datetime

class CloudUploader:
    def __init__(self):
        self.upload_results = {}
        
    def upload_to_file_io(self, file_path, filename):
        """Upload file to file.io (temporary cloud storage)"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f)}
                response = requests.post('https://file.io', files=files)
                
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return {
                        'success': True,
                        'url': result.get('link'),
                        'key': result.get('key'),
                        'expiry': '14 days',
                        'service': 'file.io'
                    }
            return {'success': False, 'error': 'Upload failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def upload_to_0x0_st(self, file_path, filename):
        """Upload file to 0x0.st (temporary cloud storage)"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f)}
                response = requests.post('https://0x0.st', files=files)
                
            if response.status_code == 200:
                url = response.text.strip()
                return {
                    'success': True,
                    'url': url,
                    'expiry': '365 days',
                    'service': '0x0.st'
                }
            return {'success': False, 'error': 'Upload failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def upload_files(self):
        """Upload both PPT and PDF files to cloud storage"""
        files_to_upload = [
            {
                'path': 'static/Technical_Seminar_Report.pdf',
                'name': 'AI_Deepfake_Detection_Technical_Report.pdf',
                'type': 'Technical Report'
            },
            {
                'path': 'static/Tech Sem PPT.pptx',
                'name': 'AI_Deepfake_Detection_Presentation.pptx',
                'type': 'Seminar Presentation'
            }
        ]
        
        results = {}
        
        for file_info in files_to_upload:
            if os.path.exists(file_info['path']):
                print(f"Uploading {file_info['type']}...")
                
                # Try multiple cloud services
                upload_result = None
                
                # Try file.io first
                result = self.upload_to_file_io(file_info['path'], file_info['name'])
                if result['success']:
                    upload_result = result
                else:
                    # Try 0x0.st as backup
                    result = self.upload_to_0x0_st(file_info['path'], file_info['name'])
                    if result['success']:
                        upload_result = result
                
                if upload_result:
                    results[file_info['type']] = upload_result
                    print(f"✅ {file_info['type']} uploaded successfully!")
                    print(f"   URL: {upload_result['url']}")
                    print(f"   Service: {upload_result['service']}")
                    print(f"   Expiry: {upload_result['expiry']}")
                else:
                    results[file_info['type']] = {'success': False, 'error': 'All upload services failed'}
                    print(f"❌ Failed to upload {file_info['type']}")
            else:
                results[file_info['type']] = {'success': False, 'error': 'File not found'}
                print(f"❌ File not found: {file_info['path']}")
        
        # Save results to JSON file
        upload_info = {
            'timestamp': datetime.now().isoformat(),
            'uploads': results
        }
        
        with open('cloud_upload_results.json', 'w') as f:
            json.dump(upload_info, f, indent=2)
        
        print(f"\n📋 Upload results saved to: cloud_upload_results.json")
        return results

def main():
    """Main function to run cloud upload"""
    print("🚀 Starting cloud upload process...")
    print("=" * 50)
    
    uploader = CloudUploader()
    results = uploader.upload_files()
    
    print("\n" + "=" * 50)
    print("📊 Upload Summary:")
    
    for file_type, result in results.items():
        if result['success']:
            print(f"✅ {file_type}: {result['url']}")
        else:
            print(f"❌ {file_type}: {result['error']}")
    
    print("\n🌐 Your files are now available in the cloud!")
    print("💡 Add these URLs to your contact page for easy sharing.")

if __name__ == "__main__":
    main()
