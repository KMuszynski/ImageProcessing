from PIL import Image
import numpy as np
import sys

from components.functions.elementary import doBrightness, doContrast, doNegative
from components.functions.geometric import doHorizontalFlip, doVerticalFlip, doDiagonalFlip, doShrink, doEnlarge
from components.functions.noise import doMedianFilter, doGeometricMeanFilter


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
            
            --median <size> [metric]   Apply median filter with the specified size and optional similarity metric
            --gmean <size> [metric]    Apply geometric mean filter with the specified size and optional similarity metric

        Similarity Metrics (optional for noise removal methods):
            mse                        Mean square error
            pmse                       Peak mean square error
            snr                        Signal to noise ratio
            psnr                       Peak signal to noise ratio
            md                         Maximum difference
        
        Examples:
            python3 program.py --brightness 50
            python3 program.py --contrast -20
            python3 program.py --negative
            python3 program.py --median 3 mse
            python3 program.py --gmean 5 psnr
            
    """
    print(help_text)


noParamFunctions = ["--negative", "--help", "--hflip", "--vflip", "--dflip"]
noiseRemovalFunctions = ["--median", "--gmean"]

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
if len(sys.argv) > 3:
    if command not in noiseRemovalFunctions:
        print("Too many command line parameters given.\n")
        print_help()
        sys.exit()

# Store the parameter if present
if len(sys.argv) == 3:
    param = sys.argv[2]

# Load the images
im = Image.open("./components/images/c_lenac_small.bmp")
arr = np.array(im)

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
    result_arr = doMedianFilter(param)
elif command == '--gmean':
    result_arr = doGeometricMeanFilter(param, arr)
else:
    print("Unknown command: " + command)
    print_help()
    sys.exit()

# Create the new image from the result array
newIm = Image.fromarray(result_arr.astype(np.uint8))
newIm.save("result.bmp")
