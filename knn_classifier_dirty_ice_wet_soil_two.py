import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from google.colab.patches import cv2_imshow

# --- Configuration ---
core_path = '/content/Utqiagvik-N20-34-38-2022.png'
# Lists allow you to add as many reference images as you like
ground_ref_paths = ['/content/ground_etalon1.png', '/content/ground_etalon2.png']
ice_ref_paths = ['/content/ice_etalon1.png', '/content/ice_etalon2.png']

def calculate_permafrost_metrics(core_path, ground_paths, ice_paths):
    # 1. Load the core image
    core = cv2.imread(core_path)
    if core is None:
        raise FileNotFoundError(f"Error: Core image not found at {core_path}")

    core_lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB)

    # 2. Extract Training Data for Ground
    X_ground_list = []
    for path in ground_paths:
        img = cv2.imread(path)
        if img is not None:
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            X_ground_list.append(img_lab.reshape(-1, 3))
    X_ground = np.vstack(X_ground_list)

    # 3. Extract Training Data for Ice
    X_ice_list = []
    for path in ice_paths:
        img = cv2.imread(path)
        if img is not None:
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            X_ice_list.append(img_lab.reshape(-1, 3))
    X_ice = np.vstack(X_ice_list)

    # 4. Prepare Labels (0 for Ground, 1 for Ice)
    X = np.vstack((X_ground, X_ice))
    y = np.hstack((np.zeros(len(X_ground)), np.ones(len(X_ice))))

    # 5. Train KNN Classifier
    # n_neighbors=5 is usually a good balance for noisy permafrost photos
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    # 6. Predict every pixel in the Core
    h, w, _ = core_lab.shape
    core_pixels = core_lab.reshape(-1, 3)
    preds = knn.predict(core_pixels)

    # 7. Calculate Metrics
    ice_count = np.sum(preds == 1)
    total_pixels = len(preds)
    ice_percent = (ice_count / total_pixels) * 100
    ground_percent = 100 - ice_percent

    # 8. Create Visual Map (Ice highlighted in Cyan)
    mask = preds.reshape(h, w)
    vis_map = core.copy()
    vis_map[mask == 1] = [255, 255, 0]

    return ice_percent, ground_percent, vis_map

# Run Analysis
ice_pct, ground_pct, visual_result = calculate_permafrost_metrics(core_path, ground_ref_paths, ice_ref_paths)

# --- OUTPUT ---
print("## METRIC ANALYSIS RESULTS")
print(f"**Total Visible Ice:** {ice_pct:.2f}%")
print(f"**Total Ground/Soil:** {ground_pct:.2f}%")
print("-" * 30)

cv2_imshow(visual_result)

# Export results
results_df = pd.DataFrame([{
    'File': 'Bylot_2019',
    'Visible_Ice_Pct': f"{ice_pct:.2f}%",
    'Soil_Pct': f"{ground_pct:.2f}%"
}])
results_df.to_csv('permafrost_metrics.csv', index=False)
