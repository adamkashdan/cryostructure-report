import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from google.colab.patches import cv2_imshow
from PIL import Image

# --- Configuration ---
# Update filenames to match your Colab file list
core_path = '/content/sample_data/Utqiagvik_2022.png' # Changed from .jpg to .png
ground_ref_path = '/content/sample_data/ground_etalon.png'
ice_ref_path = '/content/sample_data/ice_etalon.png'

def calculate_permafrost_metrics(core_path, ground_ref, ice_ref):
    # 1. Load images
    core = cv2.imread(core_path)
    ref_g = cv2.imread(ground_ref)
    ref_i = cv2.imread(ice_ref)

    # Add checks for successful image loading
    if core is None:
        raise FileNotFoundError(f"Error: Core image not found at {core_path}")
    if ref_g is None:
        raise FileNotFoundError(f"Error: Ground reference image not found at {ground_ref}")
    if ref_i is None:
        raise FileNotFoundError(f"Error: Ice reference image not found at {ice_ref}")

    # Convert to LAB color space (Better for separating "dirty ice" from "wet soil")
    core_lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB)
    ref_g_lab = cv2.cvtColor(ref_g, cv2.COLOR_BGR2LAB)
    ref_i_lab = cv2.cvtColor(ref_i, cv2.COLOR_BGR2LAB)

    # 2. Extract Training Data from Etalons
    X_ground = ref_g_lab.reshape(-1, 3)
    X_ice = ref_i_lab.reshape(-1, 3)

    # Label 0 for Ground, 1 for Ice
    X = np.vstack((X_ground, X_ice))
    y = np.hstack((np.zeros(len(X_ground)), np.ones(len(X_ice))))

    # 3. Train KNN Classifier
    # This learns the specific "signature" of ice in your photo
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    # 4. Predict every pixel in the Core
    h, w, _ = core_lab.shape
    core_pixels = core_lab.reshape(-1, 3)
    preds = knn.predict(core_pixels)

    # 5. Calculate Metrics
    ice_count = np.sum(preds == 1)
    total_pixels = len(preds)
    ice_percent = (ice_count / total_pixels) * 100
    ground_percent = 100 - ice_percent

    # 6. Create Visual Map (Ice highlighted in Cyan)
    mask = preds.reshape(h, w)
    vis_map = core.copy()
    vis_map[mask == 1] = [255, 255, 0] # Cyan/Yellow highlight for ice

    return ice_percent, ground_percent, vis_map

# Run Analysis
ice_pct, ground_pct, visual_result = calculate_permafrost_metrics(core_path, ground_ref_path, ice_ref_path)

# --- OUTPUT ---
print("## METRIC ANALYSIS RESULTS")
print(f"**Total Visible Ice:** {ice_pct:.2f}%")
print(f"**Total Ground/Soil:** {ground_pct:.2f}%")
print("-" * 30)

# Show the result image
cv2_imshow(visual_result)

# Save results for Misha
results_df = pd.DataFrame([{
    'File': 'Utqiagvik_2022',
    'Visible_Ice_Pct': f"{ice_pct:.2f}%",
    'Soil_Pct': f"{ground_pct:.2f}%"
}])
results_df.to_csv('/content/sample_data/permafrost_metrics.csv', index=False)
print("\nResults exported to permafrost_metrics.csv")
