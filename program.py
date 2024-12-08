from PIL import Image
import numpy as np
import sys

from matplotlib import pyplot as plt

from components.functions.elementary import doBrightness, doContrast, doNegative
from components.functions.geometric import doHorizontalFlip, doVerticalFlip, doDiagonalFlip, doShrink, doEnlarge
from components.functions.noise import doMedianFilter, doGeometricMeanFilter
from components.functions.histogram import calculate_save_histogram, calculate_manual_histogram
from components.functions.filtration import universal_filter, optimized_slowpass_filter
from components.functions.rayleigh import apply_rayleigh_pdf_histogram
from components.functions.laplacian import ll_operator, apply_roberts
from components.functions.statistic import (
    calculate_mean,
    calculate_variance,
    calculate_standard_deviation,
    calculate_variation_coefficient_i,
    calculate_asymmetry_coefficient,
    calculate_flattening_coefficient,
    calculate_variation_coefficient_ii,
    calculate_entropy,
)
from morphological import (
    dilation,
    erosion,
    opening,
    closing,
    hmt_transformation,
    iterative_morphological_operation,
    get_structuring_element
)


def print_help():
    help_text = """
        Usage: python3 program.py [command] [options]

        Commands:
            --help                     Show this help message

            --brightness <value>       Apply brightness adjustment (value: -100 to 100)
            --contrast <value>         Apply contrast adjustment (value: -100 to 100)
            --negative                 Apply negative effect to the image

            --hflip                    Apply horizontal flip
            --vflip                    Apply vertical flip
            --dflip                    Apply diagonal flip
            --shrink <factor>          Shrink the image by the specified factor
            --enlarge <factor>         Enlarge the image by the specified factor

            --median <size>            Apply median filter with the specified size
            --gmean <size>             Apply geometric mean filter with the specified size

            --histogram <channel>      Calculate and save the histogram for the specified channel 
                                        (options: gray, red, green, blue)

            --rayleigh [g_min] [g_max] Apply Rayleigh enhancement with optional custom g_min and g_max.
                                       If g_min and g_max are not provided, defaults are 0 and 255.

        Examples:
            python3 program.py --brightness 50
            python3 program.py --contrast -20
            python3 program.py --negative
            python3 program.py --histogram gray
            python3 program.py --rayleigh 0 200
    """
    print(help_text)


# Removed '--rayleigh' from noParamFunctions, since we now allow parameters
noParamFunctions = ["--negative", "--help", "--hflip", "--vflip", "--dflip", "--slowpass", "--grayscale-histogram", "--roberts"]

# Check if no command line parameters were given
if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    print_help()
    sys.exit()

# Store the command from the command line
command = sys.argv[1]

if command not in noParamFunctions and len(sys.argv) < 3 and command != '--histogram' and command != '--rayleigh':
    print("Too few command line parameters given.\n")
    print_help()
    sys.exit()

if command not in ['--histogram', '--rayleigh'] and len(sys.argv) > 3:
    print("Too many command line parameters given.\n")
    print_help()
    sys.exit()

param = None
if len(sys.argv) >= 3:
    param = sys.argv[2]

# Load the image
image_path = "./components/images/lena.bmp"
image = Image.open(image_path)
if image.mode not in ("RGB", "L"):
    image = image.convert("RGB")
arr = np.array(image)

B = get_structuring_element(param)

# Apply the command
if command == '--help':
    print_help()
    sys.exit()

elif command == '--brightness':
    result_arr = doBrightness(param, arr)

elif command == '--contrast':
    result_arr = doContrast(param, arr)

elif command == '--negative':
    result_arr = doNegative(arr)

elif command == '--hflip':
    result_arr = doHorizontalFlip(arr)

elif command == '--vflip':
    result_arr = doVerticalFlip(arr)

elif command == '--dflip':
    result_arr = doDiagonalFlip(arr)

elif command == '--shrink':
    result_arr = doShrink(param, arr)

elif command == '--enlarge':
    result_arr = doEnlarge(param, arr)

elif command == '--median':
    result_arr = doMedianFilter(param, arr)

elif command == '--gmean':
    result_arr = doGeometricMeanFilter(param, arr)

elif command == '--slowpass':
    result_arr = optimized_slowpass_filter(arr)

elif command == '--universal':
    result_arr = universal_filter(arr, int(param))

# Morfological
# Apply the command
elif command == '--dilation':
    result_arr = dilation(B, arr)
elif command == '--erosion':
    result_arr = erosion(B, arr)
elif command == '--opening':
    result_arr = opening(B, arr)
elif command == '--closing':
    result_arr = closing(B, arr)
elif command == '--htm':
    result_arr = hmt_transformation(B, arr)
elif command == '--morphological':
    p = (int(sys.argv[3]), int(sys.argv[4]))  # Starting point for iterative morphological operation
    result_arr = iterative_morphological_operation(B, arr, p)
# Morfological /

elif command == '--histogram':
    if param not in ["gray", "red", "green", "blue"]:
        print("Invalid channel! Choose from 'gray', 'red', 'green', or 'blue'.")
        sys.exit()
    save_path = "histogram_" + param + ".png"
    calculate_save_histogram(image_path, channel=param, save_path=save_path)
    print(f"Histogram successfully saved as {save_path}!")
    sys.exit()

elif command == '--rayleigh':
    try:
        # Default values
        g_min_val = 0
        g_max_val = 255
        q_val = 0.999

        # program.py --rayleigh [g_min] [g_max]
        if len(sys.argv) >= 3:
            g_min_val = float(sys.argv[2])
        if len(sys.argv) >= 4:
            g_max_val = float(sys.argv[3])

        # Compute alpha for printing
        alpha_val = (g_max_val - g_min_val) / np.sqrt(-2 * np.log(1 - q_val))

        # Calculate and print metrics BEFORE enhancement
        print("Metrics BEFORE enhancement:")
        print(f"Mean: {calculate_mean(arr):.2f}")
        print(f"Variance: {calculate_variance(arr):.2f}")
        print(f"Standard Deviation: {calculate_standard_deviation(arr):.2f}")
        print(f"Variation Coefficient I: {calculate_variation_coefficient_i(arr):.2f}")
        print(f"Asymmetry Coefficient: {calculate_asymmetry_coefficient(arr):.2f}")
        print(f"Flattening Coefficient: {calculate_flattening_coefficient(arr):.2f}")
        print(f"Variation Coefficient II: {calculate_variation_coefficient_ii(arr):.2f}")
        print(f"Entropy: {calculate_entropy(arr):.2f}")

        # Print chosen parameters and alpha
        print(f"\nApplying Rayleigh enhancement with:")
        print(f"  g_min = {g_min_val}")
        print(f"  g_max = {g_max_val}")
        print(f"  q = {q_val}")
        print(f"  Computed alpha = {alpha_val:.2f}")

        # Apply Rayleigh enhancement
        result_arr = apply_rayleigh_pdf_histogram(arr, g_min=g_min_val, g_max=g_max_val, q=q_val)
        print("\nImage enhanced using histogram-based Rayleigh PDF.")

        # Calculate and print metrics AFTER enhancement
        print("\nMetrics AFTER enhancement:")
        print(f"Mean: {calculate_mean(result_arr):.2f}")
        print(f"Variance: {calculate_variance(result_arr):.2f}")
        print(f"Standard Deviation: {calculate_standard_deviation(result_arr):.2f}")
        print(f"Variation Coefficient I: {calculate_variation_coefficient_i(result_arr):.2f}")
        print(f"Asymmetry Coefficient: {calculate_asymmetry_coefficient(result_arr):.2f}")
        print(f"Flattening Coefficient: {calculate_flattening_coefficient(result_arr):.2f}")
        print(f"Variation Coefficient II: {calculate_variation_coefficient_ii(result_arr):.2f}")
        print(f"Entropy: {calculate_entropy(result_arr):.2f}")

    except Exception as e:
        print(f"Error during Rayleigh enhancement: {e}")
        sys.exit()

elif command == '--grayscale-histogram':
    try:
        grayscale_image = image.convert("L")
        result_arr = np.array(grayscale_image)
        histogram = calculate_manual_histogram(result_arr)
        plt.bar(range(256), histogram, color='gray', width=1.0)
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

        print("Grayscale histogram displayed, and grayscale image saved as 'result.bmp'.")
    except Exception as e:
        print(f"Error calculating grayscale histogram: {e}")
        sys.exit()

elif command == '--oll':
    # Get the alpha parameter, default is 1.0
    alpha = float(param) if param else 1.0
    result_arr = ll_operator(arr, alpha)

elif command == '--roberts':
    try:
        result_arr = apply_roberts(arr)
        print("Roberts operator applied.")
    except Exception as e:
        print(f"Error applying Roberts operator: {e}")
        sys.exit()

else:
    print("Unknown command: " + command)
    print_help()
    sys.exit()

# Create the new image from the result array
newIm = Image.fromarray(result_arr.astype(np.uint8))
newIm.save("result.bmp")
