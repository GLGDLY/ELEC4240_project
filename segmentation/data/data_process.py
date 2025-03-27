import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import glob
from tqdm import tqdm  

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
                coords = [float(x) for x in parts[:8]]
              
                polygon = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                polygons.append(polygon)
            except ValueError:
                continue
    
    return polygons

def create_text_mask(image_size, polygons):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    
    return np.array(mask)

def process_rctw_dataset(dataset_path, output_path):
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
    
    image_files = glob.glob(os.path.join(dataset_path, "train/images/*.jpg"))
    
    print(f"Processing {len(image_files)} images...")
    
    for image_path in tqdm(image_files):
        try:
            annotation_path = image_path.replace("images", "annotations").replace(".jpg", ".txt")
            
            if not os.path.exists(annotation_path):
                print(f"Missing annotation for {image_path}")
                continue
            
            image = Image.open(image_path)
            
            polygons = parse_rctw_annotation(annotation_path)
            
            mask = create_text_mask(image.size, polygons)
            
            image_name = os.path.basename(image_path)
            mask_name = image_name.replace('.jpg', '_mask.jpg')
            
            image.save(os.path.join(output_path, "images", image_name))
            cv2.imwrite(os.path.join(output_path, "masks", mask_name), mask)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

def verify_processing(output_path, num_samples=5):
    import matplotlib.pyplot as plt
    
    image_files = os.listdir(os.path.join(output_path, "images"))
    
    for i in range(min(num_samples, len(image_files))):
        image_name = image_files[i]
        mask_name = image_name.replace('.jpg', '_mask.jpg')
        
        image = cv2.imread(os.path.join(output_path, "images", image_name))
        mask = cv2.imread(os.path.join(output_path, "masks", mask_name), 0)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title("Generated Mask")
        
        plt.subplot(133)
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]  
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay")
        
        plt.show()
