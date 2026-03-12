import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ────────────────────────────────────────────────────────────────────
MAIN_IMAGE     = "/content/Utqiagvik-N20-34-38-2022.png"
# Added multiple etalon paths
ICE_ETALONS    = ["/content/ice_etalon1.png", "/content/ice_etalon2.png"]
GROUND_ETALONS = ["/content/ground_etalon1.png", "/content/ground_etalon2.png"]

OUTPUT_FIGURE  = "/content/metric_analysis.png"
OUTPUT_DPI     = 150
# ──────────────────────────────────────────────────────────────────────────────

def load_rgb(path: str) -> np.ndarray:
    """Load an image and return it as a float32 RGB array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)

def combined_etalon_mean(paths: list) -> np.ndarray:
    """Combine multiple etalon images and return the global mean RGB."""
    all_pixels = []
    for path in paths:
        img = load_rgb(path)
        all_pixels.append(img.reshape(-1, 3))
    # Stack all pixels from all images in the list and find the mean
    return np.vstack(all_pixels).mean(axis=0)

def euclidean_dist(pixels: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance to a reference colour centroid."""
    return np.sqrt(((pixels - mean) ** 2).sum(axis=1))

def classify(main: np.ndarray, ice_mean: np.ndarray, ground_mean: np.ndarray):
    """Classify every pixel as ice or ground via nearest centroid."""
    pixels = main.reshape(-1, 3)
    dist_ice    = euclidean_dist(pixels, ice_mean)
    dist_ground = euclidean_dist(pixels, ground_mean)
    ice_mask    = (dist_ice < dist_ground).reshape(main.shape[:2])
    return ice_mask, ~ice_mask

def make_classification_rgb(main: np.ndarray, ice_mask: np.ndarray) -> np.ndarray:
    """Build a false-colour RGB image for visualization."""
    vis = np.zeros_like(main, dtype=np.uint8)
    vis[ ice_mask] = [180, 210, 255] # Light Blue
    vis[~ice_mask] = [160, 120,  80] # Brown
    return vis

def save_figure(main: np.ndarray, class_vis: np.ndarray, ice_pct: float,
                ground_pct: float, out_path: str, dpi: int = 150) -> None:
    """Render and save the 3-panel analysis figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')

    # Panel 1 – Original
    axes[0].imshow(main.astype(np.uint8))
    axes[0].set_title('Original Image', color='white', fontsize=13, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Panel 2 – Classification Map
    ice_patch    = mpatches.Patch(color=(180/255, 210/255, 1.0), label=f'Ice: {ice_pct:.1f}%')
    ground_patch = mpatches.Patch(color=(160/255, 120/255, 80/255), label=f'Ground: {ground_pct:.1f}%')
    axes[1].imshow(class_vis)
    axes[1].set_title('Classification Map', color='white', fontsize=13, fontweight='bold', pad=10)
    axes[1].axis('off')
    axes[1].legend(handles=[ice_patch, ground_patch], loc='lower left', facecolor='#1a1f2e', edgecolor='#444', labelcolor='white')

    # Panel 3 – Bar Chart
    categories = ['Ice', 'Ground']
    values     = [ice_pct, ground_pct]
    colors     = ['#64b4ff', '#a07850']
    axes[2].bar(categories, values, color=colors, width=0.5)
    axes[2].set_ylim(0, 100)
    axes[2].set_title('Metric Analysis (%)', color='white', fontsize=13, fontweight='bold')
    axes[2].set_facecolor('#1a1f2e')
    axes[2].tick_params(colors='white')

    for i, val in enumerate(values):
        axes[2].text(i, val + 1, f'{val:.1f}%', ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, facecolor='#0d1117')
    plt.close(fig)

def main():
    print("Loading images and calculating reference centroids...")
    img_main = load_rgb(MAIN_IMAGE)

    # Calculate means from multiple files
    mean_ice    = combined_etalon_mean(ICE_ETALONS)
    mean_ground = combined_etalon_mean(GROUND_ETALONS)

    print(f"  Combined Ice Mean RGB:    {mean_ice.round(1)}")
    print(f"  Combined Ground Mean RGB: {mean_ground.round(1)}")

    # Classify
    ice_mask, ground_mask = classify(img_main, mean_ice, mean_ground)
    total = ice_mask.size
    ice_pct = ice_mask.sum() / total * 100
    gnd_pct = ground_mask.sum() / total * 100

    print("\n=== METRIC ANALYSIS RESULTS ===")
    print(f"  Ice Coverage   : {ice_pct:.2f}%")
    print(f"  Ground Coverage: {gnd_pct:.2f}%")

    # Figure
    class_vis = make_classification_rgb(img_main, ice_mask)
    save_figure(img_main, class_vis, ice_pct, gnd_pct, OUTPUT_FIGURE, OUTPUT_DPI)
    print(f"\nFigure saved → {OUTPUT_FIGURE}")

if __name__ == "__main__":
    main()
