import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import sys

# MODE DEFINITIONS

def mode1(image):
    original_image = plt.imread(image)
    image_data = load_image(image)
    if image_data is None: 

        return
    
    dft_matrix = fft_2d(image_data)
    
    dft_shifted = np.fft.fftshift(dft_matrix)  # shift zero-frequency component to the center for visualization

    # plot original image
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(f'Original Image ({original_image.shape[0]}x{original_image.shape[1]})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    magnitude_spectrum = np.abs(dft_shifted)
    plt.imshow(magnitude_spectrum, cmap='gray', norm=LogNorm(vmin=1.0, vmax=magnitude_spectrum.max()))
    plt.title('Centered 2D DFT (Log Scale)')
    plt.colorbar(label='Magnitude (Log Scale)')
    plt.axis('off')
    
    plt.suptitle(f"Mode 1: Original Image and its Fourier Transform")
    plt.show()

def mode2(image):
    # load original (for display) and grayscale/padded data for processing
    try:
        original_image = plt.imread(image)
    except FileNotFoundError:
        print(f"Error: Image file '{image}' not found.")
        return

    image_data = load_image(image)
    if image_data is None:
        return

    data_for_fft = image_data.astype(float).copy()
    dft = fft_2d(data_for_fft)
    dft_shifted = np.fft.fftshift(dft) ## shift the zero frequency to center, high frequencies will be in corners

    ## filter
    rows, cols = dft_shifted.shape
    center_r  =  rows // 2
    center_c = cols // 2

    keep_frac = 0.10  ## we only keep 10% of the lowest frequencies
    radius = int(min(rows, cols) * keep_frac) ## only freqs inside this radius will be set to 1 in the mask

    r = np.arange(rows) - center_r ## array of row coordinates to compute distance later
    c = np.arange(cols) - center_c ## array of column coordinates to compute distance later
    C, R = np.meshgrid(c, r)
    dist = np.sqrt(R**2 + C**2) ## distance from the center, determines frequency magnitude
    mask = dist <= radius ## true if the point is within the radius

    dft_filtered_shifted = dft_shifted * mask ## setting all frequencies outside the radius to 0

    nonzeros = np.count_nonzero(mask)
    total = mask.size
    print(f"Using {nonzeros} non-zero Fourier coefficients out of {total}")

    dft_filtered = np.fft.ifftshift(dft_filtered_shifted) ## shift back before inverse FFT, zero frequencies at corners
    denoised_complex = ifft_2d(dft_filtered.copy()) ## inverse FFT to get denoised image
    denoised_real = np.real(denoised_complex) ## keep real part

    orig_rows, orig_cols = original_image.shape[:2] ## crop
    denoised_cropped = denoised_real[:orig_rows, :orig_cols]

    ## display
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    if original_image.ndim == 3:
        plt.imshow(original_image)
    else:
        plt.imshow(original_image, cmap='gray')
    plt.title(f'Original ({orig_rows}x{orig_cols})')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(denoised_cropped, cmap='gray')
    plt.title('Denoised (low-pass filtered)')
    plt.axis('off')
    plt.suptitle('Mode 2: Original and Denoised Image')
    plt.show()

def mode3(image):
    return

def mode4():
    return

# FOURIER ALGORITHMS

"""A 1D direct fourier transform implementation."""

def fft_1d(x):
    N = len(x)
    n = np.arange(N) ## array of 1 to N-1
    k = n.reshape((N,1)) ## column vector of frequency indices k
    e = np.exp(-2j * np.pi * k * n/N) ## matrix where element at [k,n] is the exponential for those values
    X = np.dot(e,x) ## multiply arg x with matrix
    return X


"""A 1D inverse direct fourier transform implementation."""
def ifft_1d(X):
    N = len(X)
    n = np.arange(N) ## array from 0 to N-1
    k = n.reshape((N,1)) ## frequency indices k
    e = np.exp(2j * np.pi * k * n / N) ## matrix
    x = (1/N) * np.dot(e, X)
    return x


"""A 2D FFT is performed by first applying a 1D FFT to each row, then each column."""
def fft_2d(matrix):
    ## fft on all rows
    dft_rows = np.apply_along_axis(fft_1d, axis=1, arr=matrix)

    ## fft on all columns
    dft_cols = np.apply_along_axis(fft_1d, axis=1, arr=dft_rows.T)

    ## transpose
    return dft_cols.T


'''A 2D IFFT can be implemented by performing a 1D IFFT on all the rows, 
and then performing another 1D IFFT on all the columns'''
def ifft_2d(matrix): 
    ## inverse fft on all rows
    idft_rows = np.apply_along_axis(ifft_1d, axis=1, arr=matrix)

    ## inverse fft on all columns
    idft_cols = np.apply_along_axis(ifft_1d, axis=1, arr=idft_rows.T)
    
    ## transpose
    return idft_cols.T

# """A 2D FFT is performed by first applying a 1D FFT to each column of a matrix, 
# and then applying a 1D FFT to each row of the result."""

# def fft_2d(matrix): #NOTE THIS FUNCTION MODIFIES THE INPUTED MATRIX!!!!
#     result = []
#     # 1D FFT on each column
    
#     # Get Column
#     column =[]
#     for col in range(len(matrix[0])): # assuming same size
#         for row in range(len(matrix)):
#             column.append(matrix[row][col])
#         column = fft_1d(column) 
#         for row in range(len(matrix)):
#             matrix[row][col]=column[row]
#         column =[]
    
#     # 1 D FFT on each row of result
#     for row in range(len(matrix)):
#         matrix[row] = fft_1d(matrix[row])
    
#     return matrix

# '''A 2D IFFT can be implemented by performing a 1D IFFT on all the rows of 
# the 2D frequency spectrum, and then performing another 1D IFFT on all the 
# columns of the resulting matrix'''

# def ifft_2d(matrix): 
#     result = []
#     # 1D IFFT on each column
    
#     # Get Column
#     column =[]
#     for col in range(len(matrix[0])): # assuming same size
#         for row in range(len(matrix)):
#             column.append(matrix[row][col])
#         column = ifft_1d(column) 
#         for row in range(len(matrix)):
#             matrix[row][col]=column[row]
#         column =[]
    
#     # 1 D IFFT on each row of result
#     for row in range(len(matrix)):
#         matrix[row] = ifft_1d(matrix[row])
    
#     return matrix

'''Plots the resulting 2D DFT on a log scale plot.'''

def plot_2d_dft(dft_matrix):
    mag_spec = np.log(np.abs(dft_matrix) + 1)  ## log scale 
    
    plt.imshow(mag_spec, cmap='gray')
    plt.colorbar()
    plt.title('2D DFT (Log Scale)')
    plt.show()

def is_power_of_2(n):
    return (n > 0) and ((n & (n - 1)) == 0)

def load_image(image):
    try:
        original_image = plt.imread(image)
        if original_image.ndim == 3: ## if not in 2D
            image_data = np.mean(original_image[:, :, :3], axis=2) ## convert by averaging RGB channels
        else:
            image_data = original_image
            
        rows, cols = image_data.shape
        
        if not is_power_of_2(rows) or not is_power_of_2(cols):
            print(f"Image dimensions are not powers of 2... resizing.")
            
            max_dim = max(rows, cols)
            next_power = 1
            while next_power < max_dim: ## find next power of 2 to pad wth 0s
                next_power *= 2
            
            new_image = np.zeros((next_power, next_power))
            new_image[:rows, :cols] = image_data 
            
            return new_image
        
        return image_data

    except FileNotFoundError:
        print(f"Error: Image file '{image}' not found.")
        return None
    
# COMAND LINE PARSING

print(f"Starting argument parsing...")   

parser = argparse.ArgumentParser()
parser.add_argument("-m", help="Specify a mode", type=int, choices=[1,2,3,4], default = 1)
parser.add_argument("-i", help="Specify an image", default = "moonlanding.png")

args = parser.parse_args()

# print("------ Argument Parsing Test ------")
# print(f"Mode selected (-m): {args.m}")
# print(f"Image selected (-i): {args.i}")
# print("-----------------------------------")

def main():
    print(f"Starting FFT Script in Mode {args.m} on Image: {args.i} ")   
    if args.m == 1:
        mode1(args.i)
    elif args.m == 2:
        mode2(args.i)
    elif args.m == 3:
        mode3(args.i)
    elif args.m == 4:
        mode4()

if __name__ == "__main__":
    main()
