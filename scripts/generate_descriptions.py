import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description_openai(client, image_path, max_retries=3):
    """Get detailed image description using OpenAI GPT-4 Vision."""
    base64_image = encode_image(image_path)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Provide a detailed, comprehensive description of this image. Include details about: subjects, objects, people (if any), setting, colors, mood, composition, lighting, style, and any text visible in the image. Be thorough and descriptive."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            else:
                raise e

def generate_image_descriptions(images_dir, output_json):
    """Generate descriptions for all images and save to JSON."""
    images_path = Path(images_dir)
    output_path = Path(output_json)
    
    if not images_path.exists():
        print(f"Error: Directory {images_dir} does not exist")
        return
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with: OPENAI_API_KEY=your-api-key-here")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    print("Using OpenAI GPT-4 Vision API for descriptions...\n")
    
    # Get all JPG image files
    image_files = sorted([f for f in images_path.iterdir() 
                         if f.is_file() and f.suffix.lower() == '.jpg'])
    
    print(f"Found {len(image_files)} image files to process...\n")
    
    # Process images and generate descriptions
    results = []
    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_file.name}...", end=' ', flush=True)
        
        try:
            description = get_image_description_openai(client, img_file)
            
            results.append({
                'filename': img_file.name,
                'description': description
            })
            print(f"✓ ({len(description)} chars)")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results.append({
                'filename': img_file.name,
                'description': f"Error generating description: {str(e)}"
            })
    
    # Write to JSON
    print(f"\nWriting results to {output_json}...")
    with open(output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully created {output_json} with {len(results)} image descriptions!")

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    images_dir = script_dir / 'images'
    output_json = script_dir / 'image_descriptions.json'
    
    generate_image_descriptions(images_dir, output_json)
