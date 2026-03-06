import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import pandas as pd

def calculate_visible_ice_content(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    # Convert to Grayscale to mimic the IR intensity approach used by Danielle Halle
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Enhance Contrast (Similar to the IR camera settings)
    # This helps distinguish between the 'bubbly' ground and the 'clear' ice
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 3. Thresholding (Segmentation)
    # In Halle's method, ice appears darker/different in IR. 
    # We use Otsu's thresholding to automatically find the split between ice and soil.
    _, binary_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Calculate Ice Fraction
    # Assuming Ice is the darker component in the contrast (common in core photos)
    ice_pixels = np.count_nonzero(binary_mask == 0) 
    total_pixels = binary_mask.size
    ice_content_percentage = (ice_pixels / total_pixels) * 100

    # 5. Create Visualization (similar to the 'Yellow' layers in the video at [Video IGS Halle seminar 00:41:05])
    result_vis = img.copy()
    result_vis[binary_mask == 0] = [0, 255, 255] # Highlight detected ice in Yellow

    return ice_content_percentage, result_vis

# Execute
file_path = '/content/sample_data/Utqiagvik_2022.jpg'
ice_percent, visual_map = calculate_visible_ice_content(file_path)

print(f"--- Analysis based on Halle (2023) Method ---")
print(f"Visible Ice Content: {ice_percent:.2f}%")
cv2_imshow(visual_map)
