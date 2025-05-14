# CSC 475: Image Processing - Homework 2: Image Enhancement

## Presentation Summary

### Introduction

This project demonstrates the implementation and analysis of key spatial domain image enhancement techniques discussed in Lecture 2 of CSC 475. We explored:

* **Linear Contrast Stretching:** Expanding the intensity range.
* **Gamma Correction:** Non-linear adjustment of brightness/contrast ($s=c \cdot r^{\gamma}$).
* **Histogram Equalization (HE):** Redistributing intensity levels for better contrast using the CDF.

For color images, these enhancements were applied to the Value (V) channel of the HSV color space to preserve original hues. The effects were visually compared, and the D1 histogram distance was calculated to quantify the changes in the V channel histogram between the original and enhanced images.

### Conclusion

We successfully implemented contrast stretching, gamma correction (for various gamma values), and histogram equalization. Applying these techniques to the V channel in HSV space effectively enhanced image contrast and brightness while minimizing color distortion.

Visual results and histogram analysis confirmed the expected behavior of each method:

* Contrast stretching expanded the dynamic range.
* Gamma correction controllably adjusted overall brightness (gamma < 1 brightens, gamma > 1 darkens).
* Histogram equalization significantly redistributed pixel intensities, aiming for a flatter histogram and often enhancing global contrast.

The D1 distance calculations provided a quantitative measure of how much each enhancement altered the image's V channel histogram. As expected, gamma=1.0 resulted in minimal distance, while HE and gamma values far from 1.0 showed larger distances, indicating more significant histogram changes.

This project fulfilled the homework requirements and provided practical experience with fundamental image enhancement algorithms.

## Description

This project implements and analyzes several fundamental image enhancement techniques covered in Lecture 2 of CSC 475. The goal is to understand and apply spatial domain methods for improving image quality, including contrast stretching, gamma correction, and histogram equalization. It also involves calculating image similarity based on histograms.

Based on lecture slides by Prof. Sos Agaian (CSI/CUNY).

## Homework Tasks Covered

1. **Contrast Stretching (25%):** ✅ Implementation of a linear contrast stretching algorithm.
2. **Gamma Correction (30%):** ✅ Implementation of gamma correction ($s=c \cdot r^{\gamma}$).
3. **Histogram Equalization (HE) (35%):** ✅
   * Implementation of HE using the cumulative distribution function (CDF).
   * Plotting the intensity transformation function.
   * Analysis of HE effectiveness (improvement vs. degradation) on example images.
4. **Histogram Distance (10%):** ✅ Calculation of the $D_1$ distance between the histograms of two images: $D_{1}(a,b)=\sum_{i=1}^{k}|h_{i}(a)-h_{i}(b)|$.
5. ~~(Bonus) Retinex (30%): Application of a Retinex-based enhancement method (e.g., STAR).~~

## Requirements/Dependencies

* Python 3.x
* OpenCV (`pip install opencv-python`)
* NumPy (`pip install numpy`)
* Matplotlib (`pip install matplotlib`)

All dependencies are listed in `requirements.txt`.

## How to Run

1. **Setup:**

   * Clone this repository
   * Ensure all dependencies listed in `requirements.txt` are installed:
     ```bash
     pip install -r requirements.txt
     ```
   * Place your input image in the `data/input_images/` directory
   * By default, the script will look for an image named `DSC_0165.JPG` - you can modify the `input_path` variable in `src/hw2_main.py` to use a different image
2. **Execution:** Navigate to the project directory in your terminal and run the main script:

   ```bash
   python3 src/hw2_main.py
   ```
3. **Outputs:** The script will:

   * Perform all enhancement tasks (contrast stretching, gamma correction, histogram equalization, histogram distance calculation)
   * Save output images and plots to the `results/` directory
   * Display histogram distance measurements in the console
   * Save a summary of histogram distances to `results/histogram_distances.txt`

## Results

### Task 1: Contrast Stretching

* Output: `results/cs_output.jpg`
* Comparison: `results/cs_comparison.png` (Original vs. Contrast Stretched images)

### Task 2: Gamma Correction

* Outputs with different gamma values:
  * `results/gamma_0.4_output.jpg`
  * `results/gamma_0.6_output.jpg`
  * `results/gamma_1.0_output.jpg`
  * `results/gamma_1.5_output.jpg`
  * `results/gamma_2.2_output.jpg`
* Comparison: `results/gamma_comparison.png` (Original vs. images with different gamma values)

### Task 3: Histogram Equalization

* Output: `results/he_output.jpg`
* Analysis: `results/he_analysis.png` (Original image, histogram, equalized image, equalized histogram, transformation function, CDF)
* Comparison: `results/he_comparison.png` (Original vs. our HE implementation vs. OpenCV's HE implementation)

### Task 4: Histogram Distance

* D1 distance measurements between original image and enhanced versions
* Visualization: `results/histogram_distances.png`
* Text summary: `results/histogram_distances.txt`

## Project Structure

```
hw_2/
├── data/
│   └── input_images/
│       └── DSC_0165.JPG
├── src/
│   └── hw2_main.py
├── results/
│   ├── cs_output.jpg
│   ├── cs_comparison.png
│   ├── gamma_*.jpg
│   ├── gamma_comparison.png
│   ├── he_output.jpg
│   ├── he_analysis.png
│   ├── he_comparison.png
│   ├── histogram_distances.png
│   └── histogram_distances.txt
├── readme.md
└── requirements.txt
```

## Author

* Umut Celik
