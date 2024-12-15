import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Region growing function
def region_growing(image, seed, threshold=10, max_size=1000):
    """
    Perform region growing segmentation based on pixel intensity difference.
    
    Parameters:
    - image: Input grayscale image as a numpy array.
    - seed: Starting point (x, y) for region growing.
    - threshold: Intensity difference threshold for adding neighboring pixels to the region.
    - max_size: Maximum size of the region to prevent infinite growing.
    
    Returns:
    - output: Segmented region as a binary image (0: background, 1: region).
    """
    output = np.zeros_like(image)
    rows, cols = image.shape
    region_pixels = [seed]
    output[seed] = 1
    
    while region_pixels:
        current_pixel = region_pixels.pop()
        x, y = current_pixel

        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if output[nx, ny] == 0 and abs(int(image[x, y]) - int(image[nx, ny])) <= threshold:
                        output[nx, ny] = 1
                        region_pixels.append((nx, ny))

        # Stop growing if we reach the maximum size
        if np.sum(output) >= max_size:
            break
    
    return output


def run_tests():
    """
    Run segmentation tests with different homogeneity criteria and parameters.
    """
    # Load an example image (grayscale)
    image_path = 'result.bmp'  # Replace with the actual path
    image = Image.open(image_path).convert('L')
    image = np.array(image)

    # Different homogeneity criteria (thresholds)
    thresholds = [10, 20, 30]  # Experiment with different thresholds
    seeds = [(100, 100), (50, 50), (200, 200)]  # Try different seed points
    max_region_size = 1000  # Max size of the region (optional)

    # Plot the results
    fig, axes = plt.subplots(len(thresholds), len(seeds), figsize=(12, 10))

    for i, threshold in enumerate(thresholds):
        for j, seed in enumerate(seeds):
            segmented_image = region_growing(image, seed, threshold, max_region_size)

            # Display the result
            axes[i, j].imshow(segmented_image, cmap='gray')
            axes[i, j].set_title(f"Threshold {threshold}, Seed {seed}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_tests()
