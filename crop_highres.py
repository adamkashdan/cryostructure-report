import cv2
import numpy as np

# Load the high-resolution original and the low-resolution template we want to match
high_res_path = 'Utqiagvik-N20-borecore_34-38.jpg'
template_path = 'Utqiagvik-N20-34-38-2022.png'
out_path = 'Utqiagvik-N20-34-38-2022_HighRes.png'

high_res = cv2.imread(high_res_path)
template = cv2.imread(template_path)

if high_res is None or template is None:
    print("Could not load images.")
    exit(1)

print(f"High-res shape: {high_res.shape}")
print(f"Template shape: {template.shape}")

# Convert both to grayscale for robust feature matching
gray_hr = cv2.cvtColor(high_res, cv2.COLOR_BGR2GRAY)
gray_tmp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Since the template is low-res, we cannot do direct cv2.matchTemplate.
# We need to use feature matching (SIFT/ORB) to find the bounding box, OR we can scale the template up 
# and use scale-invariant template matching. Let's use SIFT feature matching.

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray_tmp, None)
kp2, des2 = sift.detectAndCompute(gray_hr, None)

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

print(f"Found {len(good)} good feature matches.")

if len(good) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find the homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get the corners from the template image
    h, w = gray_tmp.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    
    # Project corners into the high-res image
    dst = cv2.perspectiveTransform(pts, M)

    # Find bounding box in the high-res image
    x_min = int(np.min(dst[:, 0, 0]))
    x_max = int(np.max(dst[:, 0, 0]))
    y_min = int(np.min(dst[:, 0, 1]))
    y_max = int(np.max(dst[:, 0, 1]))

    # Ensure bounds are within the image
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(high_res.shape[1], x_max)
    y_max = min(high_res.shape[0], y_max)
    
    print(f"Calculated bounding box: x=({x_min}-{x_max}), y=({y_min}-{y_max})")

    # Crop the high res image
    cropped_high_res = high_res[y_min:y_max, x_min:x_max]

    cv2.imwrite(out_path, cropped_high_res)
    print(f"Successfully saved high-resolution cropped version as: {out_path}")
    print(f"New Cropped High-Res shape: {cropped_high_res.shape}")
else:
    print("Not enough matches are found to align the images.")
