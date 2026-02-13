import numpy as np
import cv2
import PIL.Image
import PIL.ImageChops
from scipy.interpolate import griddata
import os
import threading

# --- Helper Functions ---

def RGB2gray(rgb):
    """Converts RGB image to grayscale using luminance weights."""
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# --- Error Level Analysis (ELA) ---

def get_ela(img: PIL.Image.Image, quality=90):
    """
    Generates Error Level Analysis. 
    Fixed for multi-core processing (threading) and Pillow compatibility.
    """
    thread_id = threading.get_native_id()
    tmp_file = f"temp_ela_{thread_id}.jpg"
    
    try:
        original = img.convert('RGB')
        # Save at specific quality to create compression artifacts
        original.save(tmp_file, 'JPEG', quality=quality)
        temporary = PIL.Image.open(tmp_file)
        
        # Calculate digital difference
        ela_im = PIL.ImageChops.difference(original, temporary)
        
        # Manual Scaling to brighten the heatmap
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0: 
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_im = ela_im.point(lambda p: p * scale)
        
        # Convert to numpy and normalize to [0, 1]
        result = np.array(ela_im).astype(np.float32) / 255.0
        
        # Cleanup temporary files
        temporary.close()
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
            
        return result

    except Exception as e:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise e

# --- PRNU (Noise Fingerprint) ---

def get_prnu(img: PIL.Image.Image):
    """
    Alias for noise fingerprinting to maintain compatibility with app.py imports.
    """
    return get_noise_fingerprint(img)

def get_noise_fingerprint(img: PIL.Image.Image):
    """
    Extracts high-frequency noise. Real cameras leave a 'sensor fingerprint'
    (PRNU) that AI-generated images typically lack.
    """
    img_np = np.array(img.convert('L'))
    # Use Laplacian filter to extract high-frequency noise components
    noise = cv2.Laplacian(img_np, cv2.CV_32F)
    # Normalize to [0, 1] range
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise) + 1e-8)
    return noise

# --- Texture Patching Logic ---

def img_to_patches(img: PIL.Image.Image) -> tuple:
    patch_size = 16
    img = img.convert('RGB')
    grayscale_imgs = []
    imgs = []
    coordinates = []
    # Ensure patches fit within image dimensions
    for i in range(0, img.height - patch_size + 1, patch_size):
        for j in range(0, img.width - patch_size + 1, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            img_color = np.asarray(img.crop(box))
            grayscale_image = cv2.cvtColor(src=img_color, code=cv2.COLOR_RGB2GRAY)
            grayscale_imgs.append(grayscale_image.astype(dtype=np.int32))
            imgs.append(img_color)
            normalized_coord = (i + patch_size // 2, j + patch_size // 2)
            coordinates.append(normalized_coord)
    return grayscale_imgs, imgs, coordinates, (img.height, img.width)

def get_pixel_var_degree_for_patch(patch: np.array) -> int:
    """Calculates variation degree based on neighbor differences."""
    l1 = np.sum(np.abs(patch[:, :-1] - patch[:, 1:]))
    l2 = np.sum(np.abs(patch[:-1, :] - patch[1:, :]))
    l3 = np.sum(np.abs(patch[:-1, :-1] - patch[1:, 1:]))
    l4 = np.sum(np.abs(patch[1:, :-1] - patch[:-1, 1:]))
    return l1 + l2 + l3 + l4

def get_rich_poor_patches(img: PIL.Image.Image, coloured=True):
    gray_scale_patches, color_patches, coordinates, img_size = img_to_patches(img)
    var_with_patch = []
    for i, patch in enumerate(gray_scale_patches):
        var = get_pixel_var_degree_for_patch(patch)
        target = color_patches[i] if coloured else patch
        var_with_patch.append((var, target, coordinates[i]))
    
    # Sort patches by texture richness
    var_with_patch.sort(reverse=True, key=lambda x: x[0])
    mid_point = len(var_with_patch) // 2
    
    r_patch = [(patch, coor) for var, patch, coor in var_with_patch[:mid_point]]
    p_patch = [(patch, coor) for var, patch, coor in var_with_patch[mid_point:]]
    p_patch.reverse() # Poor patches start from least variation
    return r_patch, p_patch, img_size

# --- Spectral Analysis Logic ---

def azimuthalAverage(image, center=None):
    y, x = np.indices(image.shape)
    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    r_int = r_sorted.astype(int)
    deltar = r_int[1:] - r_int[:-1]
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1]
    nr[nr == 0] = 1 # Avoid division by zero
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    radial_prof = tbin / nr
    return radial_prof

def azimuthal_integral(img, epsilon=1e-8, N=256):
    """Calculates 1D radial profile of the 2D Fourier Transform."""
    if len(img.shape) == 3:
        img = RGB2gray(img)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) + epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    psd1D = azimuthalAverage(magnitude_spectrum)
    
    # Interpolate to standardize feature vector size to N
    points = np.linspace(0, N, num=psd1D.size)
    xi = np.linspace(0, N, num=N)
    interpolated = griddata(points, psd1D, xi, method='cubic')
    interpolated = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated) + 1e-8)
    return interpolated.astype(np.float32)

def positional_emb(coor, im_size, N):
    img_height, img_width = im_size
    center_y, center_x = coor
    normalized_y = center_y / img_height
    normalized_x = center_x / img_width
    pos_emb = np.zeros(N)
    indices = np.arange(N)
    div_term = 10000 ** (2 * (indices // 2) / N)
    pos_emb[0::2] = np.sin(normalized_y / div_term[0::2]) + np.sin(normalized_x / div_term[0::2])
    pos_emb[1::2] = np.cos(normalized_y / div_term[1::2]) + np.cos(normalized_x / div_term[1::2])
    return pos_emb

# --- Primary Entry Point for preprocess.py and app.py ---

def azi_diff(img: PIL.Image.Image, patch_num=128, N=256):
    """
    Main extraction function. Combines Spectral, ELA, and PRNU features.
    Used for both training and real-time app inference.
    """
    # 1. Texture/Spectral Branches
    r, p, im_size = get_rich_poor_patches(img)
    r_len, p_len = len(r), len(p)
    
    patch_emb_r = np.zeros((patch_num, N))
    patch_emb_p = np.zeros((patch_num, N))
    pos_emb_r = np.zeros((patch_num, N))
    pos_emb_p = np.zeros((patch_num, N))

    for idx in range(patch_num):
        if r_len > 0:
            patch_emb_r[idx] = azimuthal_integral(r[idx % r_len][0], N=N)
            pos_emb_r[idx] = positional_emb(r[idx % r_len][1], im_size, N)
        if p_len > 0:
            patch_emb_p[idx] = azimuthal_integral(p[idx % p_len][0], N=N)
            pos_emb_p[idx] = positional_emb(p[idx % p_len][1], im_size, N)

    # 2. Global Visual Branches (ELA + PRNU)
    # Resizing ensures input compatibility with the CNN branches (128x128)
    ela_map = cv2.resize(get_ela(img), (128, 128))
    
    # Handle multi-channel vs single channel for PRNU
    noise_raw = get_noise_fingerprint(img)
    noise_map = cv2.resize(noise_raw, (128, 128))

    return {
        "total_emb": [patch_emb_r + pos_emb_r / 5, patch_emb_p + pos_emb_p / 5],
        "ela": ela_map, 
        "noise": noise_map,
        "image_size": im_size
    }