# ============================
# test_fft.py  (READY TO RUN)
# ============================

import numpy as np
import pytest
import time
import matplotlib.pyplot as plt 


# Import the functions from your fft.py file.
# Adjust the import names if your file uses different function names.
from fft import (
    fft_1d,
    ifft_1d,
    cooley_tuckey,
    fft2d_cooley,
    ifft_2d,
    fft_2d,
    load_image
)

# =====================================================
# 1. TEST 1D FFT / IFFT
# =====================================================

def test_fft_ifft_roundtrip_small_vectors():
    """ifft_1d(fft_1d(x)) should return x for small vectors."""
    for N in [1, 2, 4, 8, 16]:
        x = np.random.rand(N)
        x_rec = ifft_1d(fft_1d(x))
        assert np.allclose(x, x_rec, atol=1e-6)


def test_cooley_matches_numpy():
    """Cooley–Tukey implementation should match numpy.fft.fft."""
    for N in [2, 4, 8, 16, 32]:
        x = np.random.rand(N)
        assert np.allclose(cooley_tuckey(x), np.fft.fft(x), atol=1e-6)


# =====================================================
# 2. TEST 2D FFT / IFFT
# =====================================================

def test_fft2d_cooley_matches_numpy():
    """2D FFT (Cooley–Tukey) should match NumPy fft2."""
    for N in [2, 4, 8]:
        A = np.random.rand(N, N)
        assert np.allclose(fft2d_cooley(A), np.fft.fft2(A), atol=1e-6)


def test_fft2d_ifft2d_roundtrip():
    """2D FFT followed by inverse FFT should recover original matrix."""
    A = np.random.rand(16, 16)
    A_rec = ifft_2d(fft2d_cooley(A))
    assert np.allclose(A, A_rec, atol=1e-6)


def test_fft2d_naive_matches_numpy():
    """Your slower naive 2D FFT should also match NumPy for small matrices."""
    for N in [2, 4]:
        A = np.random.rand(N, N)
        assert np.allclose(fft_2d(A), np.fft.fft2(A), atol=1e-6)


# =====================================================
# 3. TEST load_image() PADDING
# =====================================================

def test_load_image_padding(tmp_path):
    """Image should be loaded and padded up to the next power of 2."""
    
    # Create a fake grayscale array (150x100)
    fake_grayscale = np.random.rand(150, 100).astype(np.float32)
    
    # Save it as a standard PNG file using matplotlib
    fpath = tmp_path / "moonlanding.png"
    plt.imsave(str(fpath), fake_grayscale, cmap='gray')

    # load_image returns (original_image_data, padded_grayscale_data)
    original, padded = load_image(str(fpath)) 

    # Next power of 2 of max(150, 100) = 150 $\rightarrow$ 256
    assert padded.shape == (256, 256)
    
    # Load the PNG back to get the *exact* array that load_image's plt.imread would get
    re_read_original = plt.imread(str(fpath))
    
    # Ensure it's treated as grayscale/2D for comparison
    if re_read_original.ndim == 3:
         re_read_original = np.mean(re_read_original[:, :, :3], axis=2)
    
    # Assert the padded area contains the re-read data
    # Use allclose due to potential minor file format conversion differences
    assert np.allclose(padded[:150, :100], re_read_original, atol=1e-5)
    
    # Check that the rest is padded with zeros
    assert np.all(padded[150:, :] == 0)
    assert np.all(padded[:, 100:] == 0)



# =====================================================
# 4. MODE 2: LOW-PASS FILTERING TEST
# =====================================================


def test_low_pass_filter_reduces_high_freq():
    """Check that low-pass filtering reduces variance of high-frequency components."""
    N = 128
    # Construct a low-frequency signal (gradient)
    low = np.linspace(0, 1, N)
    low = np.outer(low, low)

    # Construct a high-frequency noise signal (checkerboard)
    high = 0.2 * (((np.indices((N, N)).sum(axis=0)) % 2)) 
    img = low + high # Combined signal

    # --- Low-Pass Filter Logic (Similar to Mode 2) ---
    dft = fft2d_cooley(img)
    dft_shifted = np.fft.fftshift(dft)

    # Define a small radius (e.g., 5% of N) to keep only very low frequencies
    radius = int(N * 0.05) 
    center = N // 2
    r, c = np.ogrid[-center:N-center, -center:N-center]
    dist = np.sqrt(r*r + c*c)
    mask = dist <= radius # True for low freqs (circular mask)

    # Apply the mask (setting high frequencies outside the radius to 0)
    dft_filtered_shifted = dft_shifted * mask
    
    # Reconstruct
    denoised = np.real(ifft_2d(np.fft.ifftshift(dft_filtered_shifted)))

    # Check noise reduction:
    # 1. Calculate variance of the noise in the original signal
    orig_noise_var = np.var(img - low) 
    # 2. Calculate variance of the noise in the denoised signal (should be smaller)
    denoised_noise = denoised - low
    new_noise_var = np.var(denoised_noise)

    # Assert that filtering reduced the variance of the high-frequency checkerboard
    assert new_noise_var < orig_noise_var


# =====================================================
# 5. MODE 3: COMPRESSION TESTS
# =====================================================

def test_compression_threshold_percentage():
    """Check that percentile masking keeps approx the right ratio of coefficients."""
    A = np.random.rand(64, 64)
    dft = fft2d_cooley(A)
    mag = np.abs(dft)

    pct = 10  # keep 10%
    threshold = np.percentile(mag, 100 - pct)
    mask = mag >= threshold

    ratio = np.count_nonzero(mask) / mask.size
    assert abs(ratio - pct/100) < 0.02  # allow 2% error


def test_compression_monotonic_error():
    """Reconstruction error should increase as compression increases."""
    A = np.random.rand(64, 64)
    errors = []

    for pct in [100, 50, 20, 10]:
        dft = fft2d_cooley(A)
        mag = np.abs(dft)
        threshold = np.percentile(mag, 100 - pct)
        dft[mag < threshold] = 0

        A_rec = np.real(ifft_2d(dft))
        errors.append(np.mean((A - A_rec) ** 2))

    assert errors[0] < errors[1] < errors[2] < errors[3]


# =====================================================
# 6. PERFORMANCE (MODE 4) BASIC CHECKS
# =====================================================

def test_fft_runtime_increases_monotonically():
    """As N grows, FFT time must increase."""
    import time
    times = []

    for N in [16, 32, 64]:
        A = np.random.rand(N, N)
        t0 = time.time()
        fft2d_cooley(A)
        times.append(time.time() - t0)

    assert times[0] <= times[1] <= times[2]


# =====================================================
# END OF FILE
# =====================================================
