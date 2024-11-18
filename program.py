from PIL import Image
import numpy as np
import sys

from components.functions.elementary import doBrightness, doContrast, doNegative
from components.functions.geometric import doHorizontalFlip, doVerticalFlip, doDiagonalFlip, doShrink, doEnlarge
from components.functions.noise import doMedianFilter, doGeometricMeanFilter
from components.functions.histogram import calculate_save_histogram
from components.functions.filtration import universal_filter, optimized_slowpass_filter
from components.functions.rayleigh import apply_rayleigh_pdf_histogram
from components.functions.laplacian import ll_operator


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

        Examples:
            python3 program.py --brightness 50
            python3 program.py --contrast -20
            python3 program.py --negative
            python3 program.py --histogram gray        
    """
    print(help_text)


noParamFunctions = ["--negative", "--help", "--hflip", "--vflip", "--dflip",'--slowpass', "--rayleigh"]

# Check if no command line parameters were given
if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    print_help()
    sys.exit()

# Store the command from the command line
command = sys.argv[1]

# Check if there are only two arguments (program.py + command)
if len(sys.argv) == 2:
    if command not in noParamFunctions:
        print("Too few command line parameters given.\n")
        print_help()
        sys.exit()

# Check if there are more than two arguments
if len(sys.argv) > 3 and command != '--histogram':
    print("Too many command line parameters given.\n")
    print_help()
    sys.exit()

# Store param and metric if present
param = None
if len(sys.argv) >= 3:
    param = sys.argv[2]

# Load the image
image_path = "./components/images/lenac.bmp"
image = Image.open(image_path)
arr = np.array(image)

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
elif command == '--histogram':
    if param not in ["gray", "red", "green", "blue"]:
        print("Invalid channel! Choose from 'gray', 'red', 'green', or 'blue'.")
        sys.exit()
    save_path = "histogram_" + param + ".png"
    calculate_save_histogram(image_path, channel=param, save_path=save_path)
    print(f"Histogram successfully saved as {save_path}!")
    sys.exit()
elif command == '--rayleigh':
    sigma = 50
    try:
        result_arr = apply_rayleigh_pdf_histogram(arr, sigma=sigma)
        print(f"Image enhanced using histogram-based Rayleigh PDF with sigma={sigma}.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()
elif command == '--oll':
    # Get the alpha parameter, default is 1.0
    alpha = float(param) if param else 1.0
    result_arr = ll_operator(arr, alpha)
else:
    print("Unknown command: " + command)
    print_help()
    sys.exit()

# Create the new image from the result array
newIm = Image.fromarray(result_arr.astype(np.uint8))
newIm.save("result.bmp")
