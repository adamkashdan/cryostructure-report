import cv2
import numpy as np
import glob
import os

def generate_etalons_highres(core_path, mask_path, num_etalons=10, patch_size=60, step=30):
    core = cv2.imread(core_path)
    gt_color = cv2.imread(mask_path)
    
    if core is None or gt_color is None:
        print("Error: Could not load images.")
        return

    # Resize ImageJ mask to match the high-res cropped core image dimensions
    h, w = core.shape[:2]
    gt_resized = cv2.resize(gt_color, (w, h), interpolation=cv2.INTER_NEAREST)
    gt_gray = cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY)

    # In ImageJ mask: black is ice (<=127), white is ground (>127)
    gt_ice = gt_gray <= 127
    gt_ground = gt_gray > 127
    
    # Erode the masks so we don't pick up borders
    kernel = np.ones((5, 5), np.uint8)
    safely_ice = cv2.erode(gt_ice.astype(np.uint8), kernel, iterations=1) > 0
    safely_ground = cv2.erode(gt_ground.astype(np.uint8), kernel, iterations=1) > 0

    # Avoid black background of the core (the box)
    bg_mask = (core[:,:,0] < 10) & (core[:,:,1] < 10) & (core[:,:,2] < 10)

    ice_patches_info = []
    ground_patches_info = []

    for y in range(0, h - patch_size, step):
        for x in range(0, w - patch_size, step):
            patch_bg = bg_mask[y:y+patch_size, x:x+patch_size]
            if np.any(patch_bg):
                continue
                
            patch_safe_ice = safely_ice[y:y+patch_size, x:x+patch_size]
            patch_safe_ground = safely_ground[y:y+patch_size, x:x+patch_size]
            
            img_patch = core[y:y+patch_size, x:x+patch_size]
            
            # If the entire patch area is strictly inside our eroded valid masks
            if np.all(patch_safe_ice):
                ice_patches_info.append(img_patch)
            elif np.all(patch_safe_ground):
                ground_patches_info.append(img_patch)

    print(f"Candidate Ice patches: {len(ice_patches_info)}")
    print(f"Candidate Ground patches: {len(ground_patches_info)}")

    if not ice_patches_info or not ground_patches_info:
        print("Warning: Could not find enough pure patches. Trying a smaller patch size.")
        if patch_size > 10:
            return generate_etalons_highres(core_path, mask_path, num_etalons, patch_size//2, step//2)
        return

    np.random.seed(42)  # reproducible random
    
    if len(ice_patches_info) > num_etalons:
        selected_ice = [ice_patches_info[i] for i in np.random.choice(len(ice_patches_info), num_etalons, replace=False)]
    else:
        selected_ice = ice_patches_info

    if len(ground_patches_info) > num_etalons:
        selected_ground = [ground_patches_info[i] for i in np.random.choice(len(ground_patches_info), num_etalons, replace=False)]
    else:
        selected_ground = ground_patches_info

    # Delete old etalons
    for p in glob.glob('ground_etalon*.png') + glob.glob('ice_etalon*.png'):
        try:
            os.remove(p)
        except OSError:
            pass

    # Save new etalons
    for i, patch in enumerate(selected_ground, 1):
        cv2.imwrite(f'ground_etalon{i}.png', patch)
        
    for i, patch in enumerate(selected_ice, 1):
        cv2.imwrite(f'ice_etalon{i}.png', patch)
        
    print(f"Successfully saved {len(selected_ground)} ground etalons and {len(selected_ice)} ice etalons based on {core_path}.")

if __name__ == '__main__':
    generate_etalons_highres('Utqiagvik-N20-34-38-2022_HighRes.png', 'ImageJ.png', num_etalons=10, patch_size=60, step=30)
