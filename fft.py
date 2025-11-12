import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse

# MODE DEFINITIONS

def mode1(image):
    return 

def mode2(image):
    return

def mode3(image):
    return

def mode4():
    return

# FOURIER ALGORITHMS

"""A 1D FFT and IFFT implementation placeholder."""

def fft_1d(signal):
    return

def ifft_1d(signal):
    return

"""A 2D FFT is performed by first applying a 1D FFT to each column of a matrix, 
and then applying a 1D FFT to each row of the result."""

def fft_2d(matrix): #NOTE THIS FUNCTION MODIFIES THE INPUTED MATRIX!!!!
    result = []
    # 1D FFT on each column
    
    # Get Column
    column =[]
    for col in range(len(matrix[0])): # assuming same size
        for row in range(len(matrix)):
            column.append(matrix[row][col])
        column = fft_1d(column) 
        for row in range(len(matrix)):
            matrix[row][col]=column[row]
        column =[]
    
    # 1 D FFT on each row of result
    for row in range(len(matrix)):
        matrix[row] = fft_1d(matrix[row])
    
    return matrix

'''A 2D IFFT can be implemented by performing a 1D IFFT on all the rows of 
the 2D frequency spectrum, and then performing another 1D IFFT on all the 
columns of the resulting matrix'''

def ifft_2d(matrix): 
    result = []
    # 1D IFFT on each column
    
    # Get Column
    column =[]
    for col in range(len(matrix[0])): # assuming same size
        for row in range(len(matrix)):
            column.append(matrix[row][col])
        column = ifft_1d(column) 
        for row in range(len(matrix)):
            matrix[row][col]=column[row]
        column =[]
    
    # 1 D IFFT on each row of result
    for row in range(len(matrix)):
        matrix[row] = ifft_1d(matrix[row])
    
    return matrix

'''Plots the resulting 2D DFT on a log scale plot.'''

def plot_2d_dft(dft_matrix):
    mag_spec = np.log(np.abs(dft_matrix) + 1)  # Log scale for better visualization
    
    plt.imshow(mag_spec, cmap='gray')
    plt.colorbar()
    plt.title('2D DFT (Log Scale)')
    plt.show()
    
# COMAND LINE PARSING

parser = argparse.ArgumentParser()
parser.add_argument("-m", help="Specify a mode", choices=['1','2','3','4'], default = 1)
parser.add_argument("-i", help="Specify an image", default = "moonlanding.png")

args = parser.parse_args()

# print("------ Argument Parsing Test ------")
# print(f"Mode selected (-m): {args.m}")
# print(f"Image selected (-i): {args.i}")
# print("-----------------------------------")

def main():
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
