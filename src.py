import numpy as np
import matplotlib.pyplot as plt

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