from PIL import Image
import numpy as np
import sys

#############
# FUNCTIONS #
#############

def doBrightness(param, arr):
    if int(param) <= 0:
        print("error devide by zero")
        exit()
    if int(param) > 100:
        print("error brightness cant exceed 100")
        exit()
    print("Function doBrightness invoked with param: " + param)
    arr = arr * (int(param)/100)
    return arr

def doContrast(param, arr):
    contrast_factor = float(param) / 100
    pivot = 128  # Midpoint for 8-bit images

    # Calculate the new pixel values
    new_arr = pivot + contrast_factor * (arr - pivot)

    # Manually clamp values to ensure they stay within the valid range [0, 255]
    new_arr[new_arr < 0] = 0
    new_arr[new_arr > 255] = 255

    print("Function doContrast invoked with param: " + param)
    return new_arr.astype(np.uint8)

def doNegative(arr):
    arr = 255 - arr
    return arr  

def doHorizontallFlip(param, arr):
    print("This is TO BE IMPLEMENTED...")  # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doVerticalFlip(param, arr):
    print("Function doVerticalFlip invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doDiagonalFlip(param, arr):
    print("Function doDiagonalFlip invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doShrink(param, arr):
    print("Function doShrink invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doEnlarge(param, arr):
    print("Function doEnlarge invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doMedianFilter(param, arr):
    print("Function doMedianFilter invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doGeometricMeanFilter(param, arr):
    print("Function doGeometricMeanFilter invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doMeanSquareError(param, arr):
    print("Function doMeanSquareError invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doPeakMeanSquareError(param, arr):
    print("Function doPeakMeanSquareError invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doSignalToNoiseRatio(param, arr):
    print("Function doSignalToNoiseRatio invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doPeakSignalToNoiseRatio(param, arr):
    print("Function doPeakSignalToNoiseRatio invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doMaximumDifference(param, arr):
    print("Function doMaximumDifference invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

#######################
# handling parameters #
#######################
noParamFunctions = ["--negative"]

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
im = Image.open("./images/lenac_small.bmp")
arr = np.array(im.getdata())

# Reshape the array based on image channels
if arr.ndim == 1:  # Grayscale
    numColorChannels = 1
    arr = arr.reshape(im.size[1], im.size[0])
else:
    numColorChannels = arr.shape[1]
    arr = arr.reshape(im.size[1], im.size[0], numColorChannels)

# Apply the command
if command == '--brightness':
    arr = doBrightness(param, arr)
elif command == '--contrast':
    arr = doContrast(param, arr)
elif command == '--negative':
    arr = doNegative(arr)
elif command == '--hflip':
    arr = doHorizontalFlip(param, arr)
elif command == '--vflip':
    arr = doVerticalFlip(param, arr)
elif command == '--dflip':
    arr = doDiagonalFlip(param, arr)
elif command == '--shrink':
    arr = doShrink(param, arr)
elif command == '--enlarge':
    arr = doEnlarge(param, arr)
elif command == '--median':
    arr = doMedianFilter(param, arr)
elif command == '--gmean':
    arr = doGeometricMeanFilter(param, arr)
elif command == '--mse':
    arr = doMeanSquareError(param, arr, original_arr)  # Assuming you have the original array
elif command == '--pmse':
    arr = doPeakMeanSquareError(param, arr, original_arr)  # Assuming you have the original array
elif command == '--snr':
    arr = doSignalToNoiseRatio(param, arr, original_arr)  # Assuming you have the original array
elif command == '--psnr':
    arr = doPeakSignalToNoiseRatio(param, arr, original_arr)  # Assuming you have the original array
elif command == '--md':
    arr = doMaximumDifference(param, arr, original_arr)  # Assuming you have the original array
else:
    print("Unknown command: " + command)
    sys.exit()

# Create the new image from the modified array
newIm = Image.fromarray(arr.astype(np.uint8))

# Save the modified image
newIm.save("result.bmp")
