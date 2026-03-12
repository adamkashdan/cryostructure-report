import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier

# ── CONFIG ────────────────────────────────────────────────────────────────────
MAIN_IMAGE     = "/content/Utqiagvik-N20-34-38-2022.png"
ICE_ETALONS    = ["/content/ice_etalon1.png", "/content/ice_etalon2.png"]
GROUND_ETALONS = ["/content/ground_etalon1.png", "/content/ground_etalon2.png"]

OUTPUT_FIGURE  = "/content/metric_analysis_knn.png"
OUTPUT_DPI     = 150
# ──────────────────────────────────────────────────────────────────────────────

def load_rgb(path: str) -> np.ndarray:
    """Load an image and return it as a float32 RGB array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)

def prepare_training_data(ice_paths: list, ground_paths: list):
    """Collect pixels from all etalons to train the KNN model."""
    X = []
    y = []

    # Process Ice Etalons (Label 1)
    for path in ice_paths:
        img_pixels = load_rgb(path).reshape(-1, 3)
        X.append(img_pixels)
        y.append(np.ones(len(img_pixels)))

    # Process Ground Etalons (Label 0)
    for path in ground_paths:
        img_pixels = load_rgb(path).reshape(-1, 3)
        X.append(img_pixels)
        y.append(np.zeros(len(img_pixels)))

    return np.vstack(X), np.concatenate(y)

def make_classification_rgb(main_shape: tuple, ice_mask: np.ndarray) -> np.ndarray:
    """Build a false-colour RGB image for visualization."""
    vis = np.zeros((main_shape[0], main_shape[1], 3), dtype=np.uint8)
    vis[ ice_mask] = [180, 210, 255] # Light Blue for Ice
    vis[~ice_mask] = [160, 120,  80] # Brown for Ground
    return vis

def save_figure(main: np.ndarray, class_vis: np.ndarray, ice_pct: float,
                ground_pct: float, out_path: str, dpi: int = 150) -> None:
    """Render and save the 3-panel analysis figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')

    axes[0].imshow(main.astype(np.uint8))
    axes[0].set_title('Original Core Image', color='white', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    ice_patch    = mpatches.Patch(color=(180/255, 210/255, 1.0), label=f'Ice: {ice_pct:.1f}%')
    ground_patch = mpatches.Patch(color=(160/255, 120/255, 80/255), label=f'Ground: {ground_pct:.1f}%')
    axes[1].imshow(class_vis)
    axes[1].set_title('KNN Classification Map', color='white', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    axes[1].legend(handles=[ice_patch, ground_patch], loc='lower left', facecolor='#1a1f2e', labelcolor='white')

    axes[2].bar(['Ice', 'Ground'], [ice_pct, ground_pct], color=['#64b4ff', '#a07850'])
    axes[2].set_ylim(0, 100)
    axes[2].set_title('Volumetric Analysis (%)', color='white', fontsize=13, fontweight='bold')
    axes[2].set_facecolor('#1a1f2e')
    axes[2].tick_params(colors='white')

    for i, val in enumerate([ice_pct, ground_pct]):
        axes[2].text(i, val + 1, f'{val:.1f}%', ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, facecolor='#0d1117')
    plt.close(fig)

def main():
    print("Step 1: Loading etalons and training KNN model...")
    img_main = load_rgb(MAIN_IMAGE)
    X_train, y_train = prepare_training_data(ICE_ETALONS, GROUND_ETALONS)

    # Initialize KNN (n_neighbors=5 helps ignore single-pixel noise)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    print("Step 2: Classifying main image pixels (this may take a moment)...")
    h, w, _ = img_main.shape
    main_pixels = img_main.reshape(-1, 3)
    preds = knn.predict(main_pixels)

    # Calculate Metrics
    ice_mask = preds.reshape(h, w).astype(bool)
    total = ice_mask.size
    ice_pct = ice_mask.sum() / total * 100
    gnd_pct = 100 - ice_pct

    print(f"\n=== KNN ANALYSIS RESULTS ===")
    print(f"  Ice Coverage   : {ice_pct:.2f}%")
    print(f"  Ground Coverage: {gnd_pct:.2f}%")

    # Figure
    class_vis = make_classification_rgb((h, w), ice_mask)
    save_figure(img_main, class_vis, ice_pct, gnd_pct, OUTPUT_FIGURE, OUTPUT_DPI)
    print(f"\nFigure saved → {OUTPUT_FIGURE}")

if __name__ == "__main__":
    main()
