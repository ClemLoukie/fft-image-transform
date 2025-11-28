import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import sys
import time
import cv2

# MODE DEFINITIONS

def mode1(image):
    original_image, image_data = load_image(image)
    print(f"Completed image loading.")
    if image_data is None: 

        return
    
    dft_matrix = fft_2d(image_data)

#============================= REFERENCE USING NUMPY - Implemented for experiment =============================#
    # ## using built-in numpy function to compare with expected result
    # dft_matrix_numpy = np.fft.fft2(image_data)
    # magnitude_spectrum_numpy = np.abs(dft_matrix_numpy)

    # ## plot expected result
    # plt.subplot(1, 3, 3)
    # plt.imshow(magnitude_spectrum_numpy, cmap='gray', norm=LogNorm(vmin=1.0, vmax=magnitude_spectrum_numpy.max()))
    # plt.title('NumPy FFT (Reference)')
    # plt.axis('off')
#============================= END OF REFERENCE USING NUMPY =============================#

    ## plot original image
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(f'Original Image ({original_image.shape[0]}x{original_image.shape[1]})')
    plt.axis('off')
    
    ## plot fft
    plt.subplot(1, 2, 2)
    magnitude_spectrum = np.abs(dft_matrix)
    plt.imshow(magnitude_spectrum, cmap='gray', norm=LogNorm(vmin=1.0, vmax=magnitude_spectrum.max()))
    plt.title(f'Centered 2D DFT (Log Scale) ({magnitude_spectrum.shape[0]}x{magnitude_spectrum.shape[1]})')
    plt.axis('off')
    
    plt.suptitle(f"Mode 1: Original Image and its Fourier Transform")
    plt.tight_layout()
    plt.show()


def mode2(image):
    original_image, image_data = load_image(image)
    if image_data is None:
        return

    data_for_fft = image_data.astype(float).copy()
    dft = fft_2d(data_for_fft)

    #============================== CIRCLE MASK METHOD ==============================
    dft_shifted = fftshift(dft) ## shift the zero frequency to center, high frequencies will be in corners

    ## filter
    rows, cols = dft.shape
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
    #============================== END OF CIRCLE MASK METHOD ==============================

    #============================== THRESHOLD METHOD - Implemented for Experiment ==============================
    # dft_shifted = fftshift(dft)  ## shift zero-frequency to center

    # magnitude_spectrum = np.abs(dft_shifted)    # compute magnitude spectrum
    # cutoff_percentile = 5  #keep only coefficients in top X percentile
    # threshold = np.percentile(magnitude_spectrum, 100 - cutoff_percentile)

    # mask = magnitude_spectrum >= threshold    # mask: 1 where magnitude >= threshold, 0 elsewhere 
    # dft_filtered_shifted = dft_shifted * mask

    # nonzeros = np.count_nonzero(mask)
    # total = mask.size
    #============================== END OF FREQUENCY THRESHOLD METHOD ==============================

    #============================== CUTOFF METHOD - Implemented for Experiment ==============================
    # dft_shifted = fftshift(dft)           # shift zero-frequency to center
    # magnitude_spectrum = np.abs(dft_shifted)

    # cutoff_value = 40000
    # mask = magnitude_spectrum >= cutoff_value    # mask: keep coefficients whose magnitude >= cutoff_value

    # dft_filtered_shifted = dft_shifted * mask

    # nonzeros = np.count_nonzero(mask)
    # total = mask.size
    #============================== END OF FREQUENCY THRESHOLD METHOD ==============================

    print(f"Using {nonzeros} non-zero Fourier coefficients out of {total} ({nonzeros/total*100:.2f}%)")

    dft_filtered = ifftshift(dft_filtered_shifted) ## shift back before inverse FFT, zero frequencies at corners
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
    plt.title('Denoised (0.10 Low-Pass Filter)')
    plt.axis('off')
    plt.suptitle('Mode 2: Original and Denoised Image')
    plt.tight_layout()
    plt.show()

def mode3(image):
    
    # Step 0: Load original (for display)
    try:
        original_image = plt.imread(image)
    except FileNotFoundError:
        print(f"Error: Image file '{image}' not found.")
        return

    original_image, image_data = load_image(image)
    if image_data is None:
        return
    
    # STEP 1: Compute 2D DFT
    data_for_fft = image_data.astype(float).copy()  
    dft = fft_2d(data_for_fft)
    
    magnitude_spectrum = np.abs(dft)
    flattentened_magnitude = magnitude_spectrum.flatten()    
    

    # STEP 2: COMPRESSION
    #levels = [0, 50, 90, 95, 99, 99.9]  ## different compression levels from 0% to 99.9%

    # switch to percentage kept

    levels = [100, 50, 10, 5, 1, 0.01] # Percent of data we are keeping = 0% to 99.9%

    compressed_images = [] # contain the 6 compressed images

    for level in levels:
        # Get the compression

        # Method only taking top coefficent
        threshold = np.percentile(flattentened_magnitude, 100 - level)
        mask = magnitude_spectrum >= threshold  ## keep coefficients above threshold
        compressed_dft = dft * mask # This is how we compress

        print("Level " + str(100-level) + "% : Using " + str(np.count_nonzero(mask)) + " non zeros")
        
        # Inverse DFT to get compressed image
        compressed_image_complex = ifft_2d(compressed_dft.copy())
        compressed_image_real = np.real(ifft_2d(compressed_dft.copy()))
        compressed_images.append(compressed_image_real)
        
        
    # STEP 3: DISPLAY
    for i in range (len(compressed_images)):
        compressed_image = compressed_images[i]
        plt.subplot(2, 3, i + 1)
        plt.imshow(compressed_image, cmap='gray')
        plt.title(f'Level {100-levels[i]}%')
        plt.axis('off')
        
    plt.suptitle("Mode 3: Image Compression at Different Levels")
    plt.show()

def mode4():
    averages_fft = []
    std_fft = []
    averages_naive = []
    std_naive = []
    
    upper = 9
    
    for i in range (5,upper+1): #2^5 and move up to 2^10
    
        times_naive = []
        times_fft = []
        for _ in range (10):
            array = np.random.rand(2**i, 2**i) # generate random 2D array
            
            # NAIVE
            start_time = time.time()
            dft_2d(array) ## changing this to the naive dft clem please check if this is okay (used to be fft_2d)
            times_naive.append(time.time() - start_time)
            
            # FFT
            start_time = time.time()
            fft_2d(array)
            times_fft.append(time.time() - start_time)
            
        average_time_naive = np.mean(times_naive)
        average_time_fft = np.mean(times_fft)
        std_time_naive = np.std(times_naive)
        std_time_fft = np.std(times_fft)

        print(f"Size {2**i}x{2**i}")
        print(f"  Naive mean: {average_time_naive}s  variance: {std_time_naive}")
        print(f"  FFT   mean: {average_time_fft}s  variance: {std_time_fft}")
        
        averages_fft.append(average_time_fft)
        std_fft.append(std_time_fft)
        averages_naive.append(average_time_naive)
        std_naive.append(std_time_naive)
    
    # Plotting
    plt.figure()
    sizes = [2**i for i in range(5,upper+1)]
    plt.errorbar(sizes, averages_naive, yerr=2*np.array(std_naive),marker='o', label="Naive DFT (2D)")
    plt.errorbar(sizes, averages_fft, yerr=2*np.array(std_fft), marker='o', label="FFT (2D)")
    plt.legend()
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Average Runtime (seconds)')
    plt.suptitle('Mode 4: Runtime Comparison of 2D DFT vs 2D FFT')
    plt.show()

# FOURIER ALGORITHMS

"""A naive 1D direct fourier transform implementation."""
def dft_1d(x):
    N = len(x)
    n = np.arange(N) ## array of 1 to N-1
    k = n.reshape((N,1)) ## column vector of frequency indices k
    e = np.exp(-2j * np.pi * k * n/N) ## matrix where element at [k,n] is the exponential for those values
    X = np.dot(e,x) ## multiply arg x with matrix
    return X


"""A naive 1D inverse direct fourier transform implementation."""
def idft_1d(X):
    N = len(X)
    n = np.arange(N) ## array from 0 to N-1
    k = n.reshape((N,1)) ## frequency indices k
    e = np.exp(2j * np.pi * k * n / N) ## matrix
    x = (1/N) * np.dot(e, X)
    return x

'''A 1D FFT implementation using the Cooley-Tukey algorithm.'''
def fft_1d(x): ## old cooley_tuckey
    if len(x) < 2:
        return x
    even = fft_1d(x[0::2])
    odd  = fft_1d(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(len(x)) / len(x))
    return np.concatenate([ even + factor[:len(x)//2] * odd, even - factor[:len(x)//2] * odd])

'''A 1D inverse FFT implementationof Cooley-Tukey.'''
def ifft_1d(X):
    result = ifft_1d_helper(X)
    return result/len(X)

def ifft_1d_helper(X):
    N = len(X)
    if N < 2:
        return X
    even = ifft_1d_helper(X[0::2])
    odd  = ifft_1d_helper(X[1::2])
    factor = np.exp( 2j * np.pi * np.arange(N) / N ) ## positive sign
    result = np.concatenate([even + factor[:N//2] * odd, even - factor[:N//2] * odd])
    return result

'''A 2D FFT implementation using the Cooley-Tukey algorithm.'''
def fft_2d(matrix):
    # apply 1D FFT to each row
    rows_done = np.apply_along_axis(fft_1d, 1, matrix)
    # apply 1D FFT to each column
    cols_done = np.apply_along_axis(fft_1d, 1, rows_done.T)
    return cols_done.T

"""A 2D DFT is performed by first applying a 1D DFT to each row, then each column."""
def dft_2d(matrix):
    ## fft on all rows
    dft_rows = np.apply_along_axis(dft_1d, axis=1, arr=matrix)

    ## fft on all columns
    dft_cols = np.apply_along_axis(dft_1d, axis=1, arr=dft_rows.T)

    ## transpose
    return dft_cols.T


'''A 2D IFFT can be implemented by performing a 1D IFFT on all the rows, 
and then performing another 1D IFFT on all the columns'''
def ifft_2d(matrix): 
    ## inverse fft on all rows
    ifft_rows = np.apply_along_axis(ifft_1d, axis=1, arr=matrix)

    ## inverse fft on all columns
    ifft_cols = np.apply_along_axis(ifft_1d, axis=1, arr=ifft_rows.T)
    
    ## transpose
    return ifft_cols.T

'''Plots the resulting 2D DFT on a log scale plot.'''

def plot_2d_dft(dft_matrix):
    magnitude_logged = np.log(np.abs(dft_matrix) + 1)  ## log scale
    
    plt.imshow(magnitude_logged, cmap='gray')
    plt.colorbar()
    plt.title('2D DFT')
    plt.show()

'''Checks if a number is a power of 2.'''
def is_power_of_2(n):
    return (n > 0) and ((n & (n - 1)) == 0)

'''Loads an image from file, converts to grayscale and pads to next power of 2 if necessary.'''
def load_image(image):
    try:
        original_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  ## read directly as grayscale
        if original_image.ndim == 3: ## if not in 2D
            image_data = np.mean(original_image[:, :, :3], axis=2) ## convert by averaging RGB channels
        else:
            image_data = original_image
            
        rows, cols = image_data.shape

        unpadded_data = image_data.copy()

        if not is_power_of_2(rows):
            power_two_rows =  2 ** int(np.ceil(math.log2(rows)))

        if not is_power_of_2(cols):
            power_two_cols =  2 ** int(np.ceil(math.log2(cols)))
        
        if not is_power_of_2(rows) or not is_power_of_2(cols):
            print(f"Image dimensions are not powers of 2... resizing.")
            new_image = np.zeros((power_two_rows, power_two_cols))
            new_og_image = np.zeros((power_two_rows, power_two_cols))
            new_image[:rows, :cols] = image_data 
            new_og_image[:rows, :cols] = unpadded_data
            
            return new_og_image, new_image
        
        return new_og_image, image_data

    except FileNotFoundError:
        print(f"Error: Image file '{image}' not found.")
        return None, None
    
"""Shifts the zero-frequency component to the center of the spectrum."""
def fftshift(array):
    array = np.asarray(array)
    if array.ndim == 1:
        mid = array.shape[0] // 2
        return np.concatenate([array[mid:], array[:mid]]) ## shift low frequencies at the beginning, high frequencies at the end
    elif array.ndim == 2:
        rows, cols = array.shape
        row_mid = rows // 2
        col_mid = cols // 2
        return np.block([[array[row_mid:, col_mid:], array[row_mid:, :col_mid]],[array[:row_mid, col_mid:], array[:row_mid, :col_mid]]]) ## swap quadrants
    else:
        raise ValueError("fftshift only supports 1D and 2D arrays")

"""Inverse of fftshift: moves zero-frequency component back to top-left."""
def ifftshift(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        mid = (arr.shape[0] + 1) // 2  # handle odd lengths
        return np.concatenate([arr[mid:], arr[:mid]])
    elif arr.ndim == 2:
        rows, cols = arr.shape
        row_mid = (rows + 1) // 2
        col_mid = (cols + 1) // 2
        return np.block([[arr[row_mid:, col_mid:], arr[row_mid:, :col_mid]],[arr[:row_mid, col_mid:], arr[:row_mid, :col_mid]]])
    else:
        raise ValueError("ifftshift only supports 1D and 2D arrays")
    
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