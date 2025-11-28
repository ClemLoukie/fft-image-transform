
import cv2
import numpy as np
import pytest
import time
import matplotlib.pyplot as plt 

# Import the functions from your fft.py file.
from fft import (
    # 1D functions
    fft_1d,    # Now the Cooley-Tukey (fast)
    dft_1d,    # Now the Naive DFT (slow)
    idft_1d,   # Now the Inverse DFT
    # 2D functions
    fft_2d,    # Now the Fast 2D FFT (uses fft_1d)
    dft_2d,    # Now the Naive 2D DFT (uses dft_1d)
    ifft_2d,   # Now the Inverse 2D DFT
    load_image
)

# =====================================================
# 1. TEST 1D FFT / IFFT
# =====================================================

def test_fft_ifft_roundtrip_small_vectors():
    """idft_1d(fft_1d(x)) should return x for small vectors."""
    for N in [1, 2, 4, 8, 16]:
        x = np.random.rand(N)
        # Using the fast fft_1d and inverse idft_1d (naive inverse)
        x_rec = idft_1d(fft_1d(x))
        assert np.allclose(x, x_rec, atol=1e-6)

def test_dft_naive_matches_numpy():
    """The slow dft_1d should match the fast vectorized DFT (numpy)."""
    for N in [2, 4, 8, 16]:
        x = np.random.rand(N)
        assert np.allclose(dft_1d(x), np.fft.fft(x), atol=1e-6)

def test_fft_cooley_matches_numpy():
    """Your fft_1d implementation (Cooley–Tukey) should match numpy.fft.fft."""
    for N in [2, 4, 8, 16, 32]:
        x = np.random.rand(N)
        assert np.allclose(fft_1d(x), np.fft.fft(x), atol=1e-6)


# =====================================================
# 2. TEST 2D FFT / IFFT
# =====================================================

def test_fft_2d_matches_numpy():
    """Your fast 2D FFT (fft_2d) should match NumPy fft2."""
    for N in [2, 4, 8]:
        A = np.random.rand(N, N)
        # Comparing your fast 2D FFT against NumPy's
        assert np.allclose(fft_2d(A), np.fft.fft2(A), atol=1e-6)


def test_fft2d_ifft2d_roundtrip():
    """2D FFT followed by inverse FFT should recover original matrix."""
    A = np.random.rand(16, 16)
    # Using your fast fft_2d and inverse ifft_2d
    A_rec = ifft_2d(fft_2d(A))
    assert np.allclose(A, A_rec, atol=1e-6)


def test_dft_2d_naive_matches_numpy():
    """Your slow 2D DFT (dft_2d) should also match NumPy for small matrices."""
    for N in [2, 4]:
        A = np.random.rand(N, N)
        # Comparing your slow 2D DFT against NumPy's
        assert np.allclose(dft_2d(A), np.fft.fft2(A), atol=1e-6)


# =====================================================
# 3. TEST load_image() PADDING
# =====================================================

def test_load_image_padding(tmp_path):
    """Image should be loaded and padded up to the next power of 2."""
    
    # Create a fake grayscale array (150x100)
    fake_grayscale = np.random.rand(150, 100).astype(np.float32)
    
    # Save it as a standard PNG file named 'moonlanding.png' using matplotlib
    fpath = tmp_path / "moonlanding.png"
    plt.imsave(str(fpath), fake_grayscale, cmap='gray')

    # load_image returns (original_image_data, padded_grayscale_data)
    original, padded = load_image(str(fpath)) 

    # Next power of 2 of max(150, 100) = 150 $\rightarrow$ 256
    assert padded.shape == (256, 256)
    
    # Load the PNG back to get the *exact* array that load_image's plt.imread would get
    re_read_original = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
    
    # Ensure it's treated as grayscale/2D for comparison
    if re_read_original.ndim == 3:
         re_read_original = np.mean(re_read_original[:, :, :3], axis=2)
    
    # Assert the padded area contains the re-read data
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
    dft = fft_2d(img) # Use your fast fft_2d
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
    denoised = np.real(ifft_2d(np.fft.ifftshift(dft_filtered_shifted))) # Use your ifft_2d

    # Check noise reduction:
    orig_noise_var = np.var(img - low) 
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
    dft = fft_2d(A) # Use your fast fft_2d
    mag = np.abs(dft)

    pct = 10 
    threshold = np.percentile(mag, 100 - pct)
    mask = mag >= threshold

    ratio = np.count_nonzero(mask) / mask.size
    assert abs(ratio - pct/100) < 0.02 


def test_compression_monotonic_error():
    """Reconstruction error should increase as compression increases."""
    A = np.random.rand(64, 64)
    errors = []

    for pct in [100, 50, 20, 10]:
        dft = fft_2d(A) # Use your fast fft_2d
        mag = np.abs(dft)
        threshold = np.percentile(mag, 100 - pct)
        dft[mag < threshold] = 0

        A_rec = np.real(ifft_2d(dft)) # Use your idft_2d
        errors.append(np.mean((A - A_rec) ** 2))

    assert errors[0] < errors[1] < errors[2] < errors[3]


# =====================================================
# 6. PERFORMANCE (MODE 4) BASIC CHECKS
# =====================================================

def test_fft_faster_than_naive_small():
    """FFT must be significantly faster than the truly naive DFT for large enough N (N=64)."""
    
    N = 64 
    A = np.random.rand(N, N)

    # Time FFT ($O(N^2 \log N)$)
    t0 = time.time()
    fft_2d(A) # Use your fast fft_2d
    t_fft = time.time() - t0

    # Time naive (O(N^4) complexity)
    t0 = time.time()
    dft_2d(A) # Use your slow dft_2d
    t_naive = time.time() - t0

    # Assert the optimized FFT is at least 10x faster than the naive version
    assert t_fft < t_naive / 10 
    
    print(f"\nN={N} Naive: {t_naive:.6f}s, FFT: {t_fft:.6f}s")


def test_fft_runtime_increases_monotonically():
    """As N grows, FFT time must increase."""
    times = []

    for N in [16, 32, 64]:
        A = np.random.rand(N, N)
        t0 = time.time()
        fft_2d(A) # Use your fast fft_2d
        times.append(time.time() - t0)

    assert times[0] <= times[1] <= times[2]