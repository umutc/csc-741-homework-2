import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def contrast_stretching(image):
    """Applies linear contrast stretching to the V channel of an HSV image.

    Args:
        image: Input BGR image (NumPy array).

    Returns:
        The contrast-stretched BGR image (NumPy array).
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply contrast stretching to the V channel
    r_min = np.min(v)
    r_max = np.max(v)
    s_min = 0
    s_max = 255

    if r_max == r_min:
        v_stretched = v # No change if V channel is flat
    else:
        # Use float for calculations
        v_float = v.astype(np.float64)
        v_stretched_float = (v_float - r_min) * ((s_max - s_min) / (r_max - r_min)) + s_min
        # Clip and convert back to uint8
        v_stretched = np.clip(v_stretched_float, s_min, s_max).astype(image.dtype)

    # Merge the channels back
    stretched_hsv = cv2.merge([h, s, v_stretched])

    # Convert back to BGR
    stretched_image = cv2.cvtColor(stretched_hsv, cv2.COLOR_HSV2BGR)

    return stretched_image

def gamma_correction(image, gamma, c=1.0):
    """Applies gamma correction to the V channel of an HSV image.
    
    Formula: s = c * r^gamma
    where r is the input pixel value normalized to [0, 1],
    and s is the output pixel value normalized to [0, 1].
    
    Args:
        image: Input BGR image (NumPy array).
        gamma: The gamma value.
        c: The scaling constant (default: 1.0).
        
    Returns:
        The gamma-corrected BGR image (NumPy array).
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Normalize the V channel to [0, 1]
    v_normalized = v.astype(np.float64) / 255.0
    
    # Apply gamma correction formula to V channel
    v_corrected_normalized = c * np.power(v_normalized, gamma)
    
    # Scale back to [0, 255] and clip
    v_corrected = 255.0 * v_corrected_normalized
    v_corrected = np.clip(v_corrected, 0, 255)
    
    # Convert back to original data type
    v_corrected = v_corrected.astype(image.dtype)

    # Merge the channels back
    corrected_hsv = cv2.merge([h, s, v_corrected])

    # Convert back to BGR
    corrected_image = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
    
    return corrected_image

def histogram_equalization(image):
    """Applies histogram equalization to an image.
    Supports both color (HSV V-channel) and grayscale images.
    """
    # Check if the image is color (BGR) or grayscale
    is_color = len(image.shape) == 3 and image.shape[2] == 3

    if is_color:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Compute histogram for V channel
        hist, bins = np.histogram(v.flatten(), 256, [0, 256])
        # CDF
        cdf = hist.cumsum()
        # Normalize CDF to [0,255]
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_norm = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_norm = np.ma.filled(cdf_norm, 0).astype('uint8')
        # Map V channel through CDF
        v_eq = cdf_norm[v]
        # Merge and convert back to BGR
        hsv_eq = cv2.merge([h, s, v_eq])
        equalized_img = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        # Return color-equalized image and hist, cdf, transform
        return equalized_img, hist, cdf, cdf_norm

    # Grayscale image fallback
    # Compute histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    # CDF
    cdf = hist.cumsum()
    # Normalize CDF to [0,255]
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_norm = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_norm = np.ma.filled(cdf_norm, 0).astype('uint8')
    # Map image through CDF
    equalized_img = cdf_norm[image]
    # Return grayscale-equalized image and hist, cdf, transform
    return equalized_img, hist, cdf, cdf_norm

def calculate_histogram_distance(hist1, hist2):
    """Calculates the D1 distance between two histograms.
    
    Formula: D1(a,b) = sum(|h_i(a) - h_i(b)|) for i=1 to k
    
    Args:
        hist1: First histogram.
        hist2: Second histogram.
        
    Returns:
        The D1 distance between the histograms.
    """
    # Ensure the histograms are of the same length
    if len(hist1) != len(hist2):
        raise ValueError("Histograms must have the same number of bins")
    
    # Calculate the absolute differences and sum them
    d1_distance = np.sum(np.abs(hist1 - hist2))
    
    return d1_distance

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('data/input_images', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def main():
    print("Homework 2: Image Enhancement")
    
    # Ensure all necessary directories exist
    create_directories()
    
    # Load the input image
    input_path = 'data/input_images/DSC_0165.JPG'
    input_img = cv2.imread(input_path)
    if input_img is None:
        print(f"Error: Could not load image from {input_path}")
        return
    
    # Task 1: Contrast Stretching
    print("\n--- Task 1: Contrast Stretching ---")
    output_path_cs = 'results/cs_output.jpg'
    
    # Apply contrast stretching directly to the BGR image (uses HSV internally)
    stretched_img = contrast_stretching(input_img)
    
    # Save the result
    cv2.imwrite(output_path_cs, stretched_img)
    print(f"Contrast stretching applied and saved to {output_path_cs}")
    
    # Display original vs stretched images side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(stretched_img, cv2.COLOR_BGR2RGB))
    plt.title('Contrast Stretched')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/cs_comparison.png')
    print("Comparison image saved to results/cs_comparison.png")
    
    # Task 2: Gamma Correction
    print("\n--- Task 2: Gamma Correction ---")
    
    # Test different gamma values
    gamma_values = [0.4, 0.6, 1.0, 1.5, 2.2]
    
    # Create a figure with subplots for different gamma values
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Apply gamma correction with different gamma values
    for i, gamma in enumerate(gamma_values):
        # Apply gamma correction directly to the BGR image (uses HSV internally)
        corrected_img = gamma_correction(input_img, gamma)
        
        # Save the result
        output_path_gamma = f'results/gamma_{gamma:.1f}_output.jpg'
        cv2.imwrite(output_path_gamma, corrected_img)
        print(f"Gamma correction (γ={gamma:.1f}) applied and saved to {output_path_gamma}")
        
        # Add to the comparison plot
        plt.subplot(2, 3, i + 2)
        plt.imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Gamma = {gamma:.1f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/gamma_comparison.png')
    print("Gamma comparison image saved to results/gamma_comparison.png")
    
    # Task 3: Histogram Equalization
    print("\n--- Task 3: Histogram Equalization ---")
    
    # Apply histogram equalization directly to the color image (will use HSV internally)
    equalized_img, hist, cdf, transform_func = histogram_equalization(input_img)
    
    # Save the equalized image
    output_path_he = 'results/he_output.jpg'
    cv2.imwrite(output_path_he, equalized_img)
    print(f"Histogram equalization applied and saved to {output_path_he}")
    
    # Plot histograms and transformation function
    plt.figure(figsize=(15, 10))
    
    # Original image and histogram (convert to grayscale for histogram)
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    v_channel = hsv_img[:,:,2]  # Get V channel for histogram
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.hist(v_channel.flatten(), 256, range=[0, 256], color='r', alpha=0.7)
    plt.title('Original V Channel Histogram')
    plt.xlim([0, 256])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Equalized image and histogram
    hsv_equalized = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2HSV)
    v_equalized = hsv_equalized[:,:,2]  # Get V channel for histogram
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
    plt.title('Equalized Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.hist(v_equalized.flatten(), 256, range=[0, 256], color='b', alpha=0.7)
    plt.title('Equalized V Channel Histogram')
    plt.xlim([0, 256])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Transformation function
    plt.subplot(2, 3, 3)
    plt.plot(np.arange(0, 256), transform_func, 'b-', linewidth=2)
    plt.title('Transformation Function')
    plt.xlabel('Input Intensity (r)')
    plt.ylabel('Output Intensity (s)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    
    # CDF
    plt.subplot(2, 3, 6)
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalize CDF to 0-255 range
    plt.plot(np.arange(0, 256), cdf_normalized, 'g-', linewidth=2)
    plt.title('Cumulative Distribution Function (CDF)')
    plt.xlabel('Intensity')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    
    plt.tight_layout()
    plt.savefig('results/he_analysis.png')
    print("Histogram equalization analysis saved to results/he_analysis.png")
    
    # Also apply OpenCV's built-in histogram equalization for comparison
    # Create an HSV version for OpenCV equalization
    hsv_for_opencv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_for_opencv)
    v_opencv_equalized = cv2.equalizeHist(v)
    hsv_opencv_equalized = cv2.merge([h, s, v_opencv_equalized])
    opencv_equalized_img = cv2.cvtColor(hsv_opencv_equalized, cv2.COLOR_HSV2BGR)
    
    # Compare our implementation with OpenCV's
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
    plt.title('Our Histogram Equalization')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(opencv_equalized_img, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV Histogram Equalization')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/he_comparison.png')
    print("Comparison with OpenCV's implementation saved to results/he_comparison.png")
    
    # Task 4: Histogram Distance
    print("\n--- Task 4: Histogram Distance ---")
    
    # Extract V channel of original image
    hsv_original = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    v_original = hsv_original[:, :, 2]

    # Histogram for original V channel
    hist_original = cv2.calcHist([v_original], [0], None, [256], [0, 256]).flatten()
    hist_original = hist_original / np.sum(hist_original)

    # Histogram for equalized image V channel
    hsv_eq = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2HSV)
    v_eq = hsv_eq[:, :, 2]
    hist_equalized = cv2.calcHist([v_eq], [0], None, [256], [0, 256]).flatten()
    hist_equalized = hist_equalized / np.sum(hist_equalized)

    # D1 distance for histogram equalization
    d1_distance = calculate_histogram_distance(hist_original, hist_equalized)
    print(f"D1 distance between original and histogram-equalized image: {d1_distance:.4f}")

    # Compute D1 distances for all gamma values
    d1_gamma = {}
    for gamma in gamma_values:
        img_gamma = cv2.imread(f'results/gamma_{gamma:.1f}_output.jpg')
        hsv_gamma = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2HSV)
        v_gamma = hsv_gamma[:, :, 2]
        hist_gamma = cv2.calcHist([v_gamma], [0], None, [256], [0, 256]).flatten()
        hist_gamma = hist_gamma / np.sum(hist_gamma)
        dist = calculate_histogram_distance(hist_original, hist_gamma)
        d1_gamma[gamma] = dist
        print(f"D1 distance between original and gamma={gamma:.1f} image: {dist:.4f}")

    # Create a bar chart to visualize the distances
    plt.figure(figsize=(10, 6))
    labels = ['Original vs HE'] + [f'Original vs γ={gamma:.1f}' for gamma in gamma_values]
    distances = [d1_distance] + [d1_gamma[gamma] for gamma in gamma_values]
    colors = ['blue'] + ['green'] * len(gamma_values)
    plt.bar(labels, distances, color=colors)
    plt.title('Histogram Distances (D1) Between Original and Enhanced Images')
    plt.ylabel('D1 Distance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('results/histogram_distances.png')
    print("Histogram distances visualization saved to results/histogram_distances.png")

    # Save all distance measurements to a text file
    with open('results/histogram_distances.txt', 'w') as f:
        f.write("D1 Histogram Distance Measurements:\n")
        f.write(f"Original vs Histogram Equalization: {d1_distance:.4f}\n")
        for gamma in gamma_values:
            f.write(f"Original vs Gamma={gamma:.1f}: {d1_gamma[gamma]:.4f}\n")
    print("Histogram distances saved to results/histogram_distances.txt")

    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
