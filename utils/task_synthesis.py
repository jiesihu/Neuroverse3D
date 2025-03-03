import numpy as np
import numpy as np
from scipy.ndimage import zoom
import random


def process_image(image_data, function, **kwargs):
    """
    Process a batch of 3D medical images with a specific function.

    Parameters:
    - image_data: numpy array of shape (1, N, 1, 128, 128, 128) or (N, 1, 128, 128, 128)
    - function: the processing function to apply (e.g., add_noise)
    - kwargs: additional arguments to pass to the function

    Returns:
    - processed_images: numpy array with processed images
    """
    if len(image_data.shape) == 5:
        # Shape is (N, 1, 128, 128, 128), squeeze to (N, 128, 128, 128)
        image_data = image_data.squeeze(1)  # Now shape is (N, 128, 128, 128)
    
    # Apply the function (e.g., add_noise) to each image
    processed_images = np.stack([function(image, **kwargs) for image in image_data])
    return processed_images[:,np.newaxis,...]

'''
Function for adding noise
'''
def add_noise(image, noise_type="gaussian", **kwargs):
    """
    Add noise to a 3D medical image.

    Parameters:
    - image: numpy array of shape (128, 128, 128)
    - noise_type: str, type of noise to add ("gaussian", "salt_pepper", "poisson")
    - kwargs: additional arguments for noise generation

    Returns:
    - noisy_image: numpy array with added noise
    """
    if noise_type == "gaussian":
        mean = kwargs.get("mean", 0)
        std = kwargs.get("std", 0.03)
        random_flag = kwargs.get("random", False)
        if random_flag: 
            std = np.random.uniform(0.15*std, std)
        
        gaussian_noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + gaussian_noise
    
    elif noise_type == "salt_pepper":
        salt_prob = kwargs.get("salt_prob", 0.04)
        pepper_prob = kwargs.get("pepper_prob", 0.04)
        random_flag = kwargs.get("random", False)
        if random_flag: 
            salt_prob = np.random.uniform(0.25*salt_prob, salt_prob)
            pepper_prob = np.random.uniform(0.25*pepper_prob, pepper_prob)
#         print('salt_prob pepper_prob',salt_prob,pepper_prob)
        
        
        noisy_image = image.copy()
        
        range_ = image.max()-image.min()
        salt = image.max()
        pepper = image.min()
        
        # Salt noise
        num_salt = np.ceil(salt_prob * image.size)
        coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = salt
        
        # Pepper noise
        num_pepper = np.ceil(pepper_prob * image.size)
        coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = pepper
    
    elif noise_type == "poisson":
        noisy_image = np.random.poisson(image)
    
    else:
        raise ValueError("Unsupported noise type. Choose from 'gaussian', 'salt_pepper', or 'poisson'.")
    
    return_ = noisy_image
    # norm
    return_ = (return_-return_.min())/(return_.max()-return_.min())
    return return_

'''
Function for painting
'''

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, x):
    return a + x * (b - a)

def grad(hash, x, y, z):
    h = hash & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

def perlin(x, y, z, perm):
    xi, yi, zi = x.astype(int) & 255, y.astype(int) & 255, z.astype(int) & 255
    xf, yf, zf = x - xi, y - yi, z - zi
    u, v, w = fade(xf), fade(yf), fade(zf)

    aaa = perm[perm[perm[xi] + yi] + zi]
    aba = perm[perm[perm[xi] + yi + 1] + zi]
    aab = perm[perm[perm[xi] + yi] + zi + 1]
    abb = perm[perm[perm[xi] + yi + 1] + zi + 1]
    baa = perm[perm[perm[xi + 1] + yi] + zi]
    bba = perm[perm[perm[xi + 1] + yi + 1] + zi]
    bab = perm[perm[perm[xi + 1] + yi] + zi + 1]
    bbb = perm[perm[perm[xi + 1] + yi + 1] + zi + 1]

    x1 = lerp(grad(aaa, xf, yf, zf), grad(baa, xf - 1, yf, zf), u)
    x2 = lerp(grad(aba, xf, yf - 1, zf), grad(bba, xf - 1, yf - 1, zf), u)
    y1 = lerp(x1, x2, v)

    x1 = lerp(grad(aab, xf, yf, zf - 1), grad(bab, xf - 1, yf, zf - 1), u)
    x2 = lerp(grad(abb, xf, yf - 1, zf - 1), grad(bbb, xf - 1, yf - 1, zf - 1), u)
    y2 = lerp(x1, x2, w)

    return (y1 + y2) / 2

def generate_perlin_noise_3d(shape, scale):
    perm = np.arange(256)
    np.random.shuffle(perm)
    perm = np.stack([perm, perm]).flatten()

    # Generate coordinates with the desired scale
    lin = [np.linspace(0, s, s // scale, endpoint=False) for s in shape]
    x, y, z = np.meshgrid(*lin, indexing='ij')

    noise = perlin(x, y, z, perm)
    noise = zoom(noise, scale, order=1)  # Interpolate to match the desired shape
    return noise

def create_inpainting_mask(image_shape, scale=10, threshold=0):
    perlin_noise = generate_perlin_noise_3d(image_shape, scale)
    mask = perlin_noise > threshold
    return mask

def apply_inpainting_mask(image, mask):
    inpainted_image = np.copy(image)
    inpainted_image[mask] = 0  # You can choose a different inpainting strategy here
    return inpainted_image

def img_2_painting(image, scales = [8, 16],thresholds = (0.2, 0.4), **kwargs):
    scale = random.choice(scales)
    threshold = random.uniform(thresholds[0], thresholds[1])
    image_shape = image.shape
    
    # Create a random binary mask from Perlin noise
    mask = create_inpainting_mask(image_shape, scale=scale, threshold=threshold)

    # Apply the mask to the image
    inpainted_image = apply_inpainting_mask(image, mask)
    
    return inpainted_image
