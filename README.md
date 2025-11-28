# ECSE 316 ASSIGNMENT 2 - FFT IMAGE TRANSFORM

This application implements several image-processing functions using the Fast Fourier Transform (FFT). Depending on the selected mode, the program can visualize Fourier transforms, denoise images, perform compression experiments, or generate runtime analysis plots.

# Usage

The script can be ran from the command line with python fft.py [-m mode] [-i image]
Example Usage: python fft.py -m 1 -i moonlanding.png

# Arguments

-m mode: Specifies the operation mode. Options include:

    1 (default) - FFT visualization: computes the FFT and displays original + Fourier transform.

    2 - Denoising: reduces noise in the image using FFT and displays the denoised result.

    3 - Compression: Performs image compression using FFT and visualizes the compressed image.

    4 - Runtime analysis: Generates runtime analysis plots for different FFT implementations.

-i image (optional): Path to the input image file. If not provided, a default moonlanding.png image is used.

# Python Version Requirements

This program is only compatible with Python versions strictly below 3.14. Some required libraries are not compatible with Python 3.14 or later.

# Dependencies

To install the required dependencies, run: pip install numpy matplotlib opencv-python