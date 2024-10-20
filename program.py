# program.py

from PIL import Image
import numpy as np
import sys

# Import functions from other files
from components.functions.elementary import doBrightness, doContrast, doNegative
from components.functions.geometric import doHorizontalFlip, doVerticalFlip, doDiagonalFlip, doShrink, doEnlarge
from components.functions.noise import doMedianFilter, doGeometricMeanFilter
from components.functions.similarity import doMeanSquareError, doPeakMeanSquareError, doSignalToNoiseRatio, \
    doPeakSignalToNoiseRatio, doMaximumDifference


#############
# FUNCTIONS #
#############

def print_help():
    help_text = """
Usage: python3 program.py [command] [options]

Commands:
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
    --mse                      Calculate mean square error against the original image
    --pmse                     Calculate peak mean square error against the original image
    --snr                      Calculate signal to noise ratio against the original image
    --psnr                     Calculate peak signal to noise ratio against the original image
    --md                       Calculate maximum difference against the original image
    --help                     Show this help message

Examples:
    python3 program.py --brightness 50
    python3 program.py --contrast -20
    python3 program.py --negative
    """
    print(help_text)


#######################
# handling parameters #
#######################
noParamFunctions = ["--negative", "--help", "--hflip", "--vflip", "--dflip"]

# Check if no command line parameters were given
if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

# Store the command from the command line
command = sys.argv[1]

# Check if there are only two arguments (command and one parameter)
if len(sys.argv) == 2:
    if command not in noParamFunctions:
        print("Too few command line parameters given.\n")
        sys.exit()

# Check if there are more than two arguments
if len(sys.argv) > 3:
    print("Too many command line parameters given.\n")
    sys.exit()

# Store the parameter if present
if len(sys.argv) == 3:
    param = sys.argv[2]

#############################
# handle image and commands #
#############################

# Load the image
im = Image.open("./components/images/lena.bmp")
arr = np.array(im)

# Apply the command
if command == '--help':
    print_help()
elif command == '--brightness':
    arr = doBrightness(param, arr)
elif command == '--contrast':
    arr = doContrast(param, arr)
elif command == '--negative':
    arr = doNegative(arr)
elif command == '--hflip':
    arr = doHorizontalFlip(arr)
elif command == '--vflip':
    arr = doVerticalFlip(arr)
elif command == '--dflip':
    arr = doDiagonalFlip(arr)
elif command == '--shrink':
    arr = doShrink(param, arr)
elif command == '--enlarge':
    arr = doEnlarge(param, arr)
elif command == '--median':
    arr = doMedianFilter(param, arr)
elif command == '--gmean':
    arr = doGeometricMeanFilter(param, arr)
elif command == '--mse':
    arr = doMeanSquareError(param, arr)
elif command == '--pmse':
    arr = doPeakMeanSquareError(param, arr)
elif command == '--snr':
    arr = doSignalToNoiseRatio(param, arr)
elif command == '--psnr':
    arr = doPeakSignalToNoiseRatio(param, arr)
elif command == '--md':
    arr = doMaximumDifference(param, arr)
else:
    print("Unknown command: " + command)
    sys.exit()

# Create the new image from the modified array
newIm = Image.fromarray(arr.astype(np.uint8))
newIm.save("result.bmp")
