import os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure, center_of_mass, measurements
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import plot_model  # Import plot_model
import cv2  # Add this import for image processing

# Load the Keras model
model = load_model('C:/xampp/htdocs/braintumorapp/unet_model_full.h5')

# Print the model summary using Keras
# print("Model Summary:")
# model.summary()  

# Generate visualization using plot_model
try:
    plot_model(model, to_file='model_architecture_model3.png', show_shapes=True, show_layer_names=True)
    print("Model visualization saved successfully.")
except Exception as e:
    print(f"Error generating model visualization: {e}")

# Define any transformations for input preprocessing
def preprocess(image):
    image = image.resize((256, 256))  # Resize to match model input shape
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize to [0, 1]
    return image_array 

def analyze_tumor(binary_mask):
    # Find tumor pixels (black regions in the mask)
    tumor_pixels = np.where(binary_mask == 0)
    
    # Calculate area (number of tumor pixels)
    area_pixels = len(tumor_pixels[0])
    
    # Calculate approximate real-world size (assuming a typical MRI resolution)
    pixel_to_mm = 0.5  # Example: 0.5 mm per pixel
    area_mm2 = area_pixels * (pixel_to_mm ** 2)
    
    # Calculate tumor area percentage
    total_area = binary_mask.size  # Total number of pixels in the mask
    tumor_area_percentage = (area_pixels / total_area) * 100 if total_area > 0 else 0
    
    return {
        'area_pixels': area_pixels,
        'area_mm2': area_mm2,
        'tumor_area_percentage': tumor_area_percentage
    }

def detect_tumor_location(binary_mask):
    """
    Detect the location of tumor (black parts) in the binary mask
    Returns the bounding box coordinates and centroid
    """
    # Find contours of the black regions (tumor)
    tumor_pixels = np.where(binary_mask == 0)
    
    if len(tumor_pixels[0]) == 0:
        return None
    
    # Calculate bounding box
    min_row, max_row = np.min(tumor_pixels[0]), np.max(tumor_pixels[0])
    min_col, max_col = np.min(tumor_pixels[1]), np.max(tumor_pixels[1])
    
    # Calculate centroid
    centroid_y = (min_row + max_row) // 2
    centroid_x = (min_col + max_col) // 2
    
    return {
        'bbox': (min_row, min_col, max_row, max_col),
        'centroid': (centroid_x, centroid_y)
    }

def process_image(image_path, output_mask_path, output_overlay_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_array = preprocess(image)

        # Perform prediction
        output = model.predict(input_array)
        output_mask = output.squeeze()

        # Ensure output_mask is 2D
        if output_mask.ndim == 3:
            output_mask = output_mask[:, :, 0]  # Take the first channel if it's 3D

        binary_mask = (output_mask > 0.5).astype(np.uint8)  # Adjust threshold as needed

        # Debugging: Check the shape of binary_mask
        print(f"Binary mask shape: {binary_mask.shape}")

        # Ensure the structure is 2D
        structure = generate_binary_structure(2, 1)  # 2D connectivity structure

        # Morphological operations to clean up the mask
        binary_mask = binary_opening(binary_mask, structure=structure).astype(np.uint8)
        binary_mask = binary_closing(binary_mask, structure=structure).astype(np.uint8)

        # Analyze tumor properties
        tumor_props = analyze_tumor(binary_mask)
        
        # Detect tumor location
        location_info = detect_tumor_location(binary_mask)
        
        if tumor_props:
            print("\nTumor Properties:")
            print(f"Area (in pixels): {tumor_props['area_pixels']}")
            print(f"Area (in mm²): {tumor_props['area_mm2']:.2f}")
            print(f"Tumor Area Percentage: {tumor_props['tumor_area_percentage']:.2f}%")
            
            if location_info:
                print("\nTumor Location:")
                print(f"Centroid (x, y): {location_info['centroid']}")
                print(f"Bounding Box (min_row, min_col, max_row, max_col): {location_info['bbox']}")

        # Save the mask
        mask_image = Image.fromarray(binary_mask * 255)
        mask_image.save(output_mask_path)

        # Create overlay
        image = image.resize(mask_image.size)
        overlay = Image.blend(image, mask_image.convert("RGB"), alpha=0.5)
        overlay.save(output_overlay_path)

        print(f"\nProcessed and saved: {output_mask_path}, {output_overlay_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Define paths for test0002.png
input_image_path = 'C:/xampp/htdocs/braintumorapp/static/test0002.png'
output_mask_path = 'C:/xampp/htdocs/braintumorapp/static/prediction_mask_test0002.png'
output_overlay_path = 'C:/xampp/htdocs/braintumorapp/static/overlay_image_test0002.png'

# Process the image
process_image(input_image_path, output_mask_path, output_overlay_path)

