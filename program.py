from PIL import Image
import numpy as np
import sys

#############
# FUNCTIONS #
#############

def doBrightness(param, arr):
    print("Function doBrightness invoked with param: " + param)
    arr = arr / 2  # Adjust brightness; this is a simple example
    return arr

def doContrast(param, arr):
    print("Function doContrast invoked with param: " + param)
    print("This is TO BE IMPLEMENTED...")  # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

def doNegative(param, arr):
    print("This is TO BE IMPLEMENTED...")  # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

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

if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

if len(sys.argv) == 2:
    print("Too few command line parameters given.\n")
    sys.exit()

command = sys.argv[1]
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
    arr = doContrast(param, arr)  # Call doContrast and get the updated arr
else:
    print("Unknown command: " + command)
    sys.exit()

# Create the new image from the modified array
newIm = Image.fromarray(arr.astype(np.uint8))

# Save the modified image
newIm.save("result.bmp")
