import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import glob
from tqdm import tqdm
import shutil

def parse_rctw_annotation(annotation_path):
    polygons = []
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            parts = line.strip().split(',')
            if len(parts) < 8:  
                continue
                
            try:
                coords = [int(float(x)) for x in parts[:8]]
                
                polygon = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
              
                if all(x >= 0 and y >= 0 for x, y in polygon):
                    polygons.append(polygon)

            except ValueError:
                continue
    
    return polygons

def create_text_mask(image_size, polygons):
    if isinstance(image_size, Image.Image):
        image_size = image_size.size
    elif not isinstance(image_size, (tuple, list)):
        raise ValueError("image_size must be a tuple (width, height) or PIL Image")
    
    mask = Image.new('L', (int(image_size[0]), int(image_size[1])), 0)
    draw = ImageDraw.Draw(mask)
    
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    
    mask_array = np.array(mask)
    mask_array = (mask_array > 127).astype(np.uint8) * 255
    return mask_array

def process_rctw_dataset(dataset_path, output_path, target_size=None):
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
    
    image_files = glob.glob(os.path.join(dataset_path, "train/images/*.jpg"))
    
    print(f"Processing {len(image_files)} images...")
    processed = 0
    errors = 0

    for image_path in tqdm(image_files):
        try:
            annotation_path = os.path.join(
                dataset_path, 
                "train/masks", 
                os.path.basename(image_path).replace(".jpg", ".txt")
            )
            
            if not os.path.exists(annotation_path):
                print(f"Missing annotation for {image_path}")
                continue
            
            image = Image.open(image_path)
            original_size = image.size

            if target_size:
                image = image.resize(target_size, Image.BILINEAR)
                current_size = target_size
            else:
                current_size = original_size

            polygons = parse_rctw_annotation(annotation_path)
            
            if target_size:
                scale_x = target_size[0] / original_size[0]
                scale_y = target_size[1] / original_size[1]
                polygons = [[(int(x * scale_x), int(y * scale_y)) for x, y in polygon] 
                           for polygon in polygons]
                
            mask = create_text_mask(current_size, polygons)
            
            image_name = os.path.basename(image_path)
            mask_name = image_name.replace('.jpg', '_mask.jpg')
            
            image.save(os.path.join(output_path, "images", image_name))
            cv2.imwrite(os.path.join(output_path, "masks", mask_name), mask)
            
            processed += 1

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            errors += 1

    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed}")
    print(f"Errors encountered: {errors}")

def verify_processing(output_path, num_samples=5, save_visualization=True):
    import matplotlib.pyplot as plt
    
    image_files = os.listdir(os.path.join(output_path, "images"))
    
    if save_visualization:
        os.makedirs(os.path.join(output_path, "verification"), exist_ok=True)
    
    for i in range(min(num_samples, len(image_files))):
        image_name = image_files[i]
        mask_name = image_name.replace('.jpg', '_mask.jpg')
        
        image = cv2.imread(os.path.join(output_path, "images", image_name))
        mask = cv2.imread(os.path.join(output_path, "masks", mask_name), 0)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title("Generated Mask")
        plt.axis('off')
        
        plt.subplot(133)
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green overlay for text regions
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay")
        plt.axis('off')
        
        if save_visualization:
            plt.savefig(os.path.join(output_path, "verification", f"verify_{i}.png"))
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    
    current_dir = os.getcwd()
    dataset_path = os.path.join(current_dir, "data")
    output_path = os.path.join(current_dir, "processed_data")
    
    print("Looking for images in:", os.path.join(dataset_path, "train/images/*.jpg"))
    
    if not os.path.exists(os.path.join(dataset_path, "train/images")):
        print("ERROR: Image directory not found!")
        print("Expected path:", os.path.join(dataset_path, "train/images"))
    
    print("Starting processing...")
    process_rctw_dataset(
        dataset_path=dataset_path,
        output_path=output_path
    )
    
    print("Processing complete!")
    
    if os.path.exists(output_path):
        print("Starting verification...")
        verify_processing(
            output_path=output_path,
            num_samples=5,
            save_visualization=True
        )
        print("Verification complete!")
    else:
        print("No output directory found at:", output_path)
