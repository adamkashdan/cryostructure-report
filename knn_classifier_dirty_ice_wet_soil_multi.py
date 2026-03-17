import cv2
import numpy as np
import pandas as pd
import glob
import sys
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Permafrost Ice/Ground Segmentation")
    parser.add_argument('--core', type=str, required=True, help="Path to the core image to analyze (e.g., Utqiagvik-N20-34-38-2022.png)")
    parser.add_argument('--algo', type=str, default='knn', choices=['knn', 'kmeans', 'rf', 'otsu'], help="Algorithm to use for segmentation")
    parser.add_argument('--mask', type=str, default=None, help="Optional: Path to ImageJ ground-truth mask for evaluation")
    parser.add_argument('--mask-ice', type=str, default='black', choices=['black', 'white'], help="Color of ice in the mask (default: black)")
    parser.add_argument('--etalon-dir', type=str, default='.', help="Directory containing the ground_etalon*.png and ice_etalon*.png files")
    parser.add_argument('--out-csv', type=str, default='permafrost_metrics.csv', help="Output CSV path")
    return parser.parse_args()

def extract_etalon_data(etalon_dir):
    ground_paths = sorted(glob.glob(f'{etalon_dir}/ground_etalon*.png'))
    ice_paths = sorted(glob.glob(f'{etalon_dir}/ice_etalon*.png'))
    
    X_ground, X_ice = [], []
    ground_colors, ice_colors = [], []
    
    for path in ground_paths:
        img = cv2.imread(path)
        if img is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            X_ground.append(lab.reshape(-1, 3))
            ground_colors.append(np.mean(lab.reshape(-1, 3), axis=0))
            
    for path in ice_paths:
        img = cv2.imread(path)
        if img is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            X_ice.append(lab.reshape(-1, 3))
            ice_colors.append(np.mean(lab.reshape(-1, 3), axis=0))
            
    if not X_ground or not X_ice:
        raise ValueError("Could not find or load enough ground/ice etalons.")
        
    X_train = np.vstack((np.vstack(X_ground), np.vstack(X_ice)))
    y_train = np.hstack((np.zeros(sum(len(x) for x in X_ground)), np.ones(sum(len(x) for x in X_ice))))
    
    avg_ground_lab = np.mean(ground_colors, axis=0) if ground_colors else np.array([128, 128, 128])
    avg_ice_lab = np.mean(ice_colors, axis=0) if ice_colors else np.array([255, 128, 128])
    
    return X_train, y_train, avg_ground_lab, avg_ice_lab

def calculate_permafrost_metrics(args):
    core = cv2.imread(args.core)
    if core is None:
        raise FileNotFoundError(f"Error: Core image not found at {args.core}")

    core_lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB)
    
    # Valid Core Area Validation Mask (exclude black background)
    bg_mask = (core[:,:,0] < 10) & (core[:,:,1] < 10) & (core[:,:,2] < 10)
    valid_mask = ~bg_mask
    valid_area = np.sum(valid_mask)
    if valid_area == 0:
        raise ValueError("Core area is completely black or empty.")

    pixels_lab_core = core_lab[valid_mask].astype(np.float32)
    valid_preds = np.zeros(valid_area, dtype=np.uint8)
    
    if args.algo in ['knn', 'rf', 'kmeans']:
        X_train, y_train, avg_ground_lab, avg_ice_lab = extract_etalon_data(args.etalon_dir)

    print(f"Running '{args.algo.upper()}' algorithm...")

    if args.algo == 'kmeans':
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels_lab_core, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        dist_0_ice = np.linalg.norm(centers[0] - avg_ice_lab)
        dist_1_ice = np.linalg.norm(centers[1] - avg_ice_lab)
        ice_cluster_idx = 0 if dist_0_ice < dist_1_ice else 1
        valid_preds = (labels.flatten() == ice_cluster_idx).astype(np.uint8)
        
    elif args.algo == 'knn':
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        # Subsample training data to prevent MemoryError and speed up
        idx = np.random.choice(len(X_train), min(100000, len(X_train)), replace=False)
        knn.fit(X_train[idx], y_train[idx])
        valid_preds = knn.predict(pixels_lab_core).astype(np.uint8)
        
    elif args.algo == 'rf':
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
        # Subsample training data for speed
        idx = np.random.choice(len(X_train), min(200000, len(X_train)), replace=False)
        rf.fit(X_train[idx], y_train[idx])
        valid_preds = rf.predict(pixels_lab_core).astype(np.uint8)
        
    elif args.algo == 'otsu':
        L_channel = core_lab[:, :, 0]
        L_valid = L_channel[valid_mask]
        # Calculate Otsu threshold only on valid core pixels
        # Create a temp image, but threshold logic needs a 2D matrix or standard func
        # We can implement a manual Otsu over an array or apply to whole image and mask later
        _, thresh = cv2.threshold(L_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        valid_preds = (thresh[valid_mask] == 255).astype(np.uint8)

    # 6. Calculate Metrics
    ice_count = np.sum(valid_preds == 1)
    ice_percent = (ice_count / valid_area) * 100
    ground_percent = 100 - ice_percent

    # 7. Create Visual Map (Ice highlighted in Cyan)
    full_preds = np.zeros((core.shape[0], core.shape[1]), dtype=np.uint8)
    full_preds[valid_mask] = valid_preds

    vis_map = core.copy()
    vis_map[full_preds == 1] = [255, 255, 0] # Cyan highlight

    return ice_percent, ground_percent, full_preds, valid_mask, vis_map, core

def evaluate_with_mask(full_preds, valid_mask, core_shape, mask_path, mask_ice_color):
    gt = cv2.imread(mask_path)
    if gt is None:
         print(f"Warning: Could not load ground truth mask {mask_path}")
         return None
         
    h, w = core_shape[:2]
    gt_resized = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    gt_gray = cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY)
    
    if mask_ice_color == 'black':
         gt_ice = (gt_gray <= 127).astype(np.uint8)
    else:
         gt_ice = (gt_gray > 127).astype(np.uint8)
         
    valid_gt_ice = gt_ice[valid_mask]
    valid_preds_ice = full_preds[valid_mask]
    
    valid_area = np.sum(valid_mask)
    gt_ice_pct = np.sum(valid_gt_ice) / valid_area * 100
    
    correct_pixels = np.sum(valid_preds_ice == valid_gt_ice)
    accuracy = correct_pixels / valid_area * 100
    
    intersection = np.sum((valid_preds_ice == 1) & (valid_gt_ice == 1))
    union = np.sum((valid_preds_ice == 1) | (valid_gt_ice == 1))
    iou = (intersection / union * 100) if union > 0 else 0
    
    return {'gt_ice_pct': gt_ice_pct, 'accuracy': accuracy, 'iou': iou}

def main():
    args = parse_args()
    
    try:
        ice_pct, ground_pct, full_preds, valid_mask, vis_map, core = calculate_permafrost_metrics(args)

        print("-" * 40)
        print("## METRIC ANALYSIS RESULTS")
        print(f"**Image Analyzed:** {args.core}")
        print(f"**Algorithm Used:** {args.algo.upper()}")
        print(f"**Total Visible Ice (Valid Core Area):** {ice_pct:.2f}%")
        print(f"**Total Ground/Soil (Valid Core Area):** {ground_pct:.2f}%")
        
        # Ground Truth Evaluation
        if args.mask:
            eval_metrics = evaluate_with_mask(full_preds, valid_mask, core.shape, args.mask, args.mask_ice)
            if eval_metrics:
                 print("-" * 40)
                 print("## GROUND TRUTH EVALUATION")
                 print(f"**Mask Used:** {args.mask} (Ice = {args.mask_ice})")
                 print(f"**Actual Ice Pct (Valid Core Area):** {eval_metrics['gt_ice_pct']:.2f}%")
                 print(f"**Difference:** {abs(ice_pct - eval_metrics['gt_ice_pct']):.2f}%")
                 print(f"**Pixel Accuracy:** {eval_metrics['accuracy']:.2f}%")
                 print(f"**Ice IoU (Intersection over Union):** {eval_metrics['iou']:.2f}%")

        print("-" * 40)

        # Output results to CSV
        results = {
            'File': args.core,
            'Algorithm': args.algo.upper(),
            'Visible_Ice_Pct': f"{ice_pct:.2f}%",
            'Soil_Pct': f"{ground_pct:.2f}%"
        }
        pd.DataFrame([results]).to_csv(args.out_csv, index=False)
        print(f"Results saved to {args.out_csv}")

        # Try to save visualization headless
        cv2.imwrite(f'automated_result_{args.algo}.jpg', vis_map)
        print(f"Saved visual result to automated_result_{args.algo}.jpg.")

    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

# python knn.py --core Utqiagvik-N20-34-38-2022_HighRes.png --algo knn --mask ImageJ.png --mask-ice black   
# python knn.py --core Utqiagvik-N20-34-38-2022_HighRes.png
