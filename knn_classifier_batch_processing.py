import cv2
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from google.colab.patches import cv2_imshow

# --- Configuration ---
folder_path = '/content/borehole_photos/'  # Folder with your 50-100 photos
ground_ref_path = '/content/ground_bench_mark.png'
ice_ref_path = '/content/ice_bench_mark.png'
output_folder = '/content/analysis_results/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- 1. Train the 'Halle-Style' Classifier Once ---
def train_classifier(g_path, i_path):
    ref_g = cv2.imread(g_path)
    ref_i = cv2.imread(i_path)
    
    # Convert to LAB (Lightness, A, B) to handle different lighting
    X_ground = cv2.cvtColor(ref_g, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    X_ice = cv2.cvtColor(ref_i, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    
    X = np.vstack((X_ground, X_ice))
    y = np.hstack((np.zeros(len(X_ground)), np.ones(len(X_ice))))
    
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X, y)
    return knn

knn_model = train_classifier(ground_ref_path, ice_ref_path)

# --- 2. Batch Processing Loop ---
all_data = []
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Starting batch process for {len(image_files)} images...")

for filename in image_files:
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # Predict Pixels
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    preds = knn_model.predict(img_lab)
    mask = preds.reshape(h, w)
    
    # Calculate Total Visible Ice %
    ice_percent = (np.sum(preds == 1) / len(preds)) * 100
    
    # Optional: Stratigraphy (Top half vs Bottom half ice %)
    top_half = preds.reshape(h, w)[:h//2, :]
    bottom_half = preds.reshape(h, w)[h//2:, :]
    ice_top = (np.sum(top_half == 1) / top_half.size) * 100
    ice_bottom = (np.sum(bottom_half == 1) / bottom_half.size) * 100

    # Save Results to List
    all_data.append({
        'Filename': filename,
        'Total_Ice_Percent': round(ice_percent, 2),
        'Ice_Top_Section': round(ice_top, 2),
        'Ice_Bottom_Section': round(ice_bottom, 2)
    })
    
    # Save a visual 'Mask' image for verification
    vis = img.copy()
    vis[mask == 1] = [255, 255, 0] # Highlight ice in Cyan
    cv2.imwrite(os.path.join(output_folder, f"analyzed_{filename}"), vis)

# --- 3. Export to CSV ---
df = pd.DataFrame(all_data)
df.to_csv('Borehole_Analysis_Summary.csv', index=False)

print("--- BATCH COMPLETE ---")
print(df.head()) # Preview the table
