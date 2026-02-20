"""
Micro Cryostructure Analysis
Permafrost Core Image Analysis Tool

Analyzes cryostructure types from permafrost core samples:
- Micro-layered lenticular
- Micro-lenticular 
- Micro-suspended

Locations: Yukon Territory (Canada), Galbraith Lake (Alaska), Willow Unit NPR-A
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import filters, measure, morphology, feature
from skimage.color import rgb2gray
import pandas as pd
from typing import Dict, Tuple, List
import seaborn as sns

# Set plotting style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'sans-serif'


class CryostructureAnalyzer:
    """Analyzer for permafrost micro cryostructure images"""
    
    def __init__(self, image_path: str):
        """
        Initialize analyzer with image path
        
        Parameters:
        -----------
        image_path : str
            Path to the cryostructure image
        """
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Metadata
        self.metadata = {
            'A': {
                'type': 'Micro-layered lenticular',
                'location': 'Yukon Territory, Canada (near Beaver Creek)',
                'depth': '127-131 cm',
                'photographer': 'M. Andersen'
            },
            'B': {
                'type': 'Micro-lenticular',
                'location': 'Galbraith Lake, Alaska',
                'depth': '186-190 cm',
                'photographer': 'M. Andersen'
            },
            'C': {
                'type': 'Micro-suspended',
                'location': 'Willow Unit, NPR-A',
                'depth': '89-93 cm',
                'photographer': 'M. Sousa'
            }
        }
        
    def split_panels(self) -> Dict[str, np.ndarray]:
        """Split the composite image into three panels (A, B, C)"""
        height, width = self.image_rgb.shape[:2]
        panel_width = width // 3
        
        panels = {
            'A': self.image_rgb[:, :panel_width, :],
            'B': self.image_rgb[:, panel_width:2*panel_width, :],
            'C': self.image_rgb[:, 2*panel_width:, :]
        }
        
        return panels
    
    def extract_texture_features(self, panel: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features from a cryostructure panel
        
        Parameters:
        -----------
        panel : np.ndarray
            Image panel to analyze
            
        Returns:
        --------
        features : dict
            Dictionary of texture features
        """
        gray_panel = rgb2gray(panel)
        
        # 1. Gradient magnitude (edge detection)
        gradient = filters.sobel(gray_panel)
        
        # 2. Local Binary Pattern (LBP) for texture
        # Simplified texture measure using variance
        texture_variance = ndimage.generic_filter(gray_panel, np.var, size=15)
        
        # 3. Fourier transform for periodicity
        f_transform = np.fft.fft2(gray_panel)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 4. Statistical features
        features = {
            'mean_intensity': np.mean(gray_panel),
            'std_intensity': np.std(gray_panel),
            'gradient_mean': np.mean(gradient),
            'gradient_std': np.std(gradient),
            'texture_variance_mean': np.mean(texture_variance),
            'texture_variance_std': np.std(texture_variance),
            'entropy': measure.shannon_entropy(gray_panel),
        }
        
        # 5. Layering detection (horizontal structure)
        # Analyze horizontal vs vertical gradients
        sobelx = cv2.Sobel(gray_panel, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_panel, cv2.CV_64F, 0, 1, ksize=5)
        
        features['horizontal_gradient'] = np.mean(np.abs(sobely))
        features['vertical_gradient'] = np.mean(np.abs(sobelx))
        features['layering_ratio'] = features['horizontal_gradient'] / (features['vertical_gradient'] + 1e-8)
        
        return features
    
    def detect_ice_lenses(self, panel: np.ndarray, threshold_factor: float = 1.2) -> Tuple[np.ndarray, int]:
        """
        Detect ice lenses/inclusions in the cryostructure
        
        Parameters:
        -----------
        panel : np.ndarray
            Image panel to analyze
        threshold_factor : float
            Threshold multiplier for detection
            
        Returns:
        --------
        labeled_image : np.ndarray
            Labeled regions
        num_lenses : int
            Number of detected lenses
        """
        gray_panel = rgb2gray(panel)
        
        # Detect bright regions (ice lenses are typically brighter)
        threshold = filters.threshold_otsu(gray_panel) * threshold_factor
        binary = gray_panel > threshold
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(binary, min_size=50)
        
        # Label connected regions
        labeled = measure.label(cleaned)
        num_lenses = labeled.max()
        
        return labeled, num_lenses
    
    def analyze_color_distribution(self, panel: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze color distribution in the panel
        
        Parameters:
        -----------
        panel : np.ndarray
            RGB image panel
            
        Returns:
        --------
        color_stats : dict
            Color channel statistics
        """
        color_stats = {}
        
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            color_stats[channel] = {
                'mean': np.mean(panel[:, :, i]),
                'std': np.std(panel[:, :, i]),
                'histogram': np.histogram(panel[:, :, i], bins=50, range=(0, 255))[0]
            }
        
        return color_stats
    
    def visualize_comprehensive_analysis(self):
        """Create comprehensive visualization of all three panels"""
        
        panels = self.split_panels()
        
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)
        
        # Row 1: Original images
        for idx, (label, panel) in enumerate(panels.items()):
            ax = fig.add_subplot(gs[0, idx])
            ax.imshow(panel)
            ax.set_title(f'{label}: {self.metadata[label]["type"]}\n'
                        f'{self.metadata[label]["location"]}\n'
                        f'Depth: {self.metadata[label]["depth"]}',
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # Add scale bar (5 mm reference)
            scale_bar_length = int(panel.shape[1] * 0.15)
            ax.plot([20, 20 + scale_bar_length], 
                   [panel.shape[0] - 30, panel.shape[0] - 30],
                   'w-', linewidth=4)
            ax.text(20 + scale_bar_length/2, panel.shape[0] - 45, '5 mm',
                   color='white', fontsize=9, ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Row 2: Grayscale and edge detection
        for idx, (label, panel) in enumerate(panels.items()):
            gray_panel = rgb2gray(panel)
            edges = filters.sobel(gray_panel)
            
            ax = fig.add_subplot(gs[1, idx])
            ax.imshow(gray_panel, cmap='gray')
            ax.set_title(f'{label}: Grayscale', fontweight='bold')
            ax.axis('off')
        
        # Row 3: Edge detection
        for idx, (label, panel) in enumerate(panels.items()):
            gray_panel = rgb2gray(panel)
            edges = filters.sobel(gray_panel)
            
            ax = fig.add_subplot(gs[2, idx])
            ax.imshow(edges, cmap='hot')
            ax.set_title(f'{label}: Edge Detection (Sobel)', fontweight='bold')
            ax.axis('off')
        
        # Row 4: Ice lens detection
        for idx, (label, panel) in enumerate(panels.items()):
            labeled, num_lenses = self.detect_ice_lenses(panel)
            
            ax = fig.add_subplot(gs[3, idx])
            ax.imshow(labeled, cmap='nipy_spectral')
            ax.set_title(f'{label}: Ice Lenses Detected\n(n={num_lenses})', fontweight='bold')
            ax.axis('off')
        
        # Row 5: Color histograms
        for idx, (label, panel) in enumerate(panels.items()):
            ax = fig.add_subplot(gs[4, idx])
            
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist, bins = np.histogram(panel[:, :, i], bins=50, range=(0, 255))
                ax.plot(bins[:-1], hist, color=color, alpha=0.7, label=color.upper())
            
            ax.set_title(f'{label}: Color Distribution', fontweight='bold')
            ax.set_xlabel('Intensity')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Comprehensive Micro Cryostructure Analysis\n'
                    'Permafrost Core Samples from Arctic Regions',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        return fig
    
    def compare_texture_features(self) -> pd.DataFrame:
        """
        Compare texture features across all three panels
        
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with feature comparison
        """
        panels = self.split_panels()
        
        feature_data = []
        for label, panel in panels.items():
            features = self.extract_texture_features(panel)
            features['Panel'] = label
            features['Type'] = self.metadata[label]['type']
            features['Location'] = self.metadata[label]['location']
            feature_data.append(features)
        
        df = pd.DataFrame(feature_data)
        return df
    
    def create_feature_comparison_plot(self, features_df: pd.DataFrame):
        """Create visualization comparing features across panels"""
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        key_features = [
            'mean_intensity', 'gradient_mean', 'texture_variance_mean',
            'entropy', 'layering_ratio', 'std_intensity'
        ]
        
        feature_labels = [
            'Mean Intensity', 'Edge Strength', 'Texture Variance',
            'Entropy', 'Layering Ratio\n(Horizontal/Vertical)', 'Intensity Std Dev'
        ]
        
        for idx, (feature, label) in enumerate(zip(key_features, feature_labels)):
            ax = axes[idx]
            
            values = features_df[feature].values
            panels = features_df['Panel'].values
            colors = ['#D4A574', '#8B7D6B', '#6B8E9D']  # Match image colors
            
            bars = ax.bar(panels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.set_title(label, fontweight='bold', fontsize=11)
            ax.set_ylabel('Value', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add cryostructure type labels
            ax.set_xticklabels([f"{p}\n{features_df[features_df['Panel']==p]['Type'].values[0][:15]}" 
                                for p in panels], fontsize=8)
        
        fig.suptitle('Quantitative Cryostructure Feature Comparison',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> str:
        """Generate text report of the analysis"""
        
        panels = self.split_panels()
        features_df = self.compare_texture_features()
        
        report = []
        report.append("="*70)
        report.append("MICRO CRYOSTRUCTURE ANALYSIS REPORT")
        report.append("="*70)
        report.append("")
        
        for label, panel in panels.items():
            meta = self.metadata[label]
            features = features_df[features_df['Panel'] == label].iloc[0]
            labeled, num_lenses = self.detect_ice_lenses(panel)
            
            report.append(f"\n{'='*70}")
            report.append(f"PANEL {label}: {meta['type'].upper()}")
            report.append(f"{'='*70}")
            report.append(f"Location: {meta['location']}")
            report.append(f"Depth: {meta['depth']}")
            report.append(f"Photographer: {meta['photographer']}")
            report.append(f"\nTEXTURE FEATURES:")
            report.append(f"  Mean Intensity: {features['mean_intensity']:.3f}")
            report.append(f"  Standard Deviation: {features['std_intensity']:.3f}")
            report.append(f"  Edge Strength: {features['gradient_mean']:.3f}")
            report.append(f"  Texture Variance: {features['texture_variance_mean']:.3f}")
            report.append(f"  Shannon Entropy: {features['entropy']:.3f}")
            report.append(f"\nSTRUCTURAL FEATURES:")
            report.append(f"  Horizontal Gradient: {features['horizontal_gradient']:.3f}")
            report.append(f"  Vertical Gradient: {features['vertical_gradient']:.3f}")
            report.append(f"  Layering Ratio: {features['layering_ratio']:.3f}")
            report.append(f"  Detected Ice Lenses: {num_lenses}")
            
            # Interpretation
            report.append(f"\nINTERPRETATION:")
            if features['layering_ratio'] > 1.5:
                report.append(f"  Strong horizontal layering detected - consistent with {meta['type']}")
            if num_lenses > 50:
                report.append(f"  High density of ice inclusions")
            if features['entropy'] > 7:
                report.append(f"  High structural complexity")
        
        report.append(f"\n{'='*70}")
        report.append("COMPARATIVE SUMMARY")
        report.append(f"{'='*70}")
        
        # Find extremes
        most_layered = features_df.loc[features_df['layering_ratio'].idxmax()]
        most_complex = features_df.loc[features_df['entropy'].idxmax()]
        
        report.append(f"\nMost Layered Structure: Panel {most_layered['Panel']} "
                     f"({most_layered['Type']})")
        report.append(f"Most Complex Structure: Panel {most_complex['Panel']} "
                     f"({most_complex['Type']})")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


def main():
    """Main execution function"""
    
    # Initialize analyzer
    print("Loading cryostructure image...")
    analyzer = CryostructureAnalyzer('/mnt/user-data/uploads/saj270182-fig-0005-m.jpg')
    
    # Generate comprehensive visualization
    print("\nGenerating comprehensive analysis...")
    fig1 = analyzer.visualize_comprehensive_analysis()
    plt.savefig('/home/claude/cryostructure_comprehensive.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: cryostructure_comprehensive.png")
    
    # Compare features
    print("\nExtracting and comparing features...")
    features_df = analyzer.compare_texture_features()
    
    # Create feature comparison plot
    fig2 = analyzer.create_feature_comparison_plot(features_df)
    plt.savefig('/home/claude/cryostructure_features.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: cryostructure_features.png")
    
    # Save feature data
    features_df.to_csv('/home/claude/cryostructure_features.csv', index=False)
    print("✓ Saved: cryostructure_features.csv")
    
    # Generate report
    print("\nGenerating analysis report...")
    report = analyzer.generate_report()
    
    with open('/home/claude/cryostructure_report.txt', 'w') as f:
        f.write(report)
    print("✓ Saved: cryostructure_report.txt")
    
    # Print report to console
    print("\n" + report)
    
    # Display feature table
    print("\n" + "="*70)
    print("FEATURE COMPARISON TABLE")
    print("="*70)
    print(features_df[['Panel', 'Type', 'mean_intensity', 'gradient_mean', 
                       'layering_ratio', 'entropy']].to_string(index=False))
    
    print("\n" + "="*70)
    print("Analysis complete! All files saved.")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
                                 
