import os
import glob

def test_paths():
    print("Current directory:", os.getcwd())
    
    data_dir = os.path.join(os.getcwd(), "data")
    print("Data directory exists:", os.path.exists(data_dir))
    
    images_dir = os.path.join(data_dir, "train", "images")
    print("Images directory exists:", os.path.exists(images_dir))
    
    image_pattern = os.path.join(images_dir, "*.jpg")
    image_files = glob.glob(image_pattern)
    print("Number of jpg files found:", len(image_files))
    if len(image_files) > 0:
        print("First few image files:", image_files[:3])
    
    masks_dir = os.path.join(data_dir, "train", "masks")
    print("Masks directory exists:", os.path.exists(masks_dir))
    
    mask_pattern = os.path.join(masks_dir, "*.txt")
    mask_files = glob.glob(mask_pattern)
    print("Number of txt files found:", len(mask_files))
    if len(mask_files) > 0:
        print("First few mask files:", mask_files[:3])

if __name__ == "__main__":
    test_paths()