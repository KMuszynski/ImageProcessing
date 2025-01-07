import numpy as np
import argparse
from PIL import Image
from test_fourier import dft_2d, idft_2d, fftshift_custom


def create_filter(filter_type, size, width, height, params={}):
    """
    Create a frequency-domain filter mask.
    
    Parameters:
        filter_type: Type of the filter (lowpass, highpass, bandpass, bandcut, edge, phase).
        size: Cutoff size (radius for low/high-pass, band size for band filters).
        width, height: Dimensions of the image.
        params: Additional parameters like edge direction or phase (k, l).
    
    Returns:
        Filter mask of shape (height, width).
    """
    mask = np.zeros((height, width), dtype=np.complex128)
    cx, cy = height // 2, width // 2  # Center of the frequency domain

    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - cy) ** 2 + (y - cx) ** 2)

    if filter_type == "lowpass":
        mask[distance <= size] = 1
    elif filter_type == "highpass":
        mask[distance > size] = 1
    elif filter_type == "bandpass":
        lower, upper = params.get("lower", 0), params.get("upper", size)
        mask[(distance > lower) & (distance <= upper)] = 1
    elif filter_type == "bandcut":
        lower, upper = params.get("lower", 0), params.get("upper", size)
        mask[(distance <= lower) | (distance > upper)] = 1
    elif filter_type == "edge":
        direction = params.get("direction", "horizontal")
        if direction == "horizontal":
            mask[cx-size:cx+size+1, :] = 1
        elif direction == "vertical":
            mask[:, cy-size:cy+size+1] = 1
    elif filter_type == "phase":
        k, l = params.get("k", 0), params.get("l", 0)
        n = np.arange(height).reshape(-1, 1)
        m = np.arange(width).reshape(1, -1)
        mask = np.exp(1j * (-n * k * 2 * np.pi / height - m * l * 2 * np.pi / width + (k + l) * np.pi))

    return mask


def apply_filter(image, filter_type, size, params={}):
    height, width, _ = image.shape

    # Apply the DFT to each RGB channel separately
    f_transform_r = dft_2d(image[:, :, 0])  # Red channel
    f_transform_g = dft_2d(image[:, :, 1])  # Green channel
    f_transform_b = dft_2d(image[:, :, 2])  # Blue channel

    # Shift zero frequency to the center of the frequency domain
    f_transform_r = fftshift_custom(f_transform_r)
    f_transform_g = fftshift_custom(f_transform_g)
    f_transform_b = fftshift_custom(f_transform_b)

    # Create the frequency domain filter mask
    mask = create_filter(filter_type, size, width, height, params)

    # Apply the mask to each frequency domain of the channels
    filtered_transform_r = f_transform_r * mask
    filtered_transform_g = f_transform_g * mask
    filtered_transform_b = f_transform_b * mask

    # Shift back the filtered transforms to original frequency domain (inverse of fftshift)
    filtered_transform_r = fftshift_custom(filtered_transform_r)
    filtered_transform_g = fftshift_custom(filtered_transform_g)
    filtered_transform_b = fftshift_custom(filtered_transform_b)

    # Apply Inverse DFT to each channel
    filtered_r = idft_2d(filtered_transform_r)
    filtered_g = idft_2d(filtered_transform_g)
    filtered_b = idft_2d(filtered_transform_b)

    # Take the real part of the result (imaginary parts should be negligible)
    filtered_r_real = np.real(filtered_r)
    filtered_g_real = np.real(filtered_g)
    filtered_b_real = np.real(filtered_b)

    # Stack the RGB channels back together to form the color image
    filtered_image_rgb = np.stack((filtered_r_real, filtered_g_real, filtered_b_real), axis=-1)

    # Clip values to ensure they are within the valid range [0, 255] for uint8
    filtered_image_rgb = np.clip(filtered_image_rgb, 0, 255).astype(np.uint8)

    return filtered_image_rgb

def main():
    parser = argparse.ArgumentParser(description="Apply frequency domain filters to an image.")
    parser.add_argument("--lowpass", type=int, help="Apply low-pass filter with specified cutoff size.")
    parser.add_argument("--highpass", type=int, help="Apply high-pass filter with specified cutoff size.")
    parser.add_argument("--bandpass", type=str, help="Apply band-pass filter with lower,upper cutoff (e.g., 10,50).")
    parser.add_argument("--bandcut", type=str, help="Apply band-cut filter with lower,upper cutoff (e.g., 10,50).")
    parser.add_argument("--edge", type=str, help="Apply edge-detection high-pass filter. Specify direction (horizontal or vertical).")
    parser.add_argument("--phase", type=str, help="Apply phase-modifying filter. Specify k,l values (e.g., 1,2).")
    parser.add_argument("--output", type=str, help="Path to save the filtered image.", default="filtered_image.bmp")
    args = parser.parse_args()

    # Load the image in RGB mode
    image_path = "./components/images/c_lenac_small.bmp"
    image = Image.open(image_path).convert("RGB")  # Convert to RGB if not already
    image_array = np.array(image)

    # Apply the specified filter
    filtered_image = None
    if args.lowpass:
        filtered_image = apply_filter(image_array, "lowpass", args.lowpass)
    elif args.highpass:
        filtered_image = apply_filter(image_array, "highpass", args.highpass)
    elif args.bandpass:
        lower, upper = map(int, args.bandpass.split(","))
        filtered_image = apply_filter(image_array, "bandpass", 0, params={"lower": lower, "upper": upper})
    elif args.bandcut:
        lower, upper = map(int, args.bandcut.split(","))
        filtered_image = apply_filter(image_array, "bandcut", 0, params={"lower": lower, "upper": upper})
    elif args.edge:
        filtered_image = apply_filter(image_array, "edge", 10, params={"direction": args.edge})
    elif args.phase:
        k, l = map(int, args.phase.split(","))
        filtered_image = apply_filter(image_array, "phase", 0, params={"k": k, "l": l})
    else:
        print("Error: No filter specified.")
        return

    # Save the filtered image
    output_path = args.output
    filtered_image_pil = Image.fromarray(filtered_image)
    filtered_image_pil.save(output_path)
    print(f"Filtered image saved to {output_path}")

if __name__ == "__main__":
    main()
