from PIL import Image
import numpy as np


def create_centered_rectangle_bmp(image_size=(100, 100), rect_size=(30, 30)):

    array = np.zeros(image_size, dtype=np.uint8)

    # Calculate the top-left corner of the rectangle
    start_x = (image_size[0] - rect_size[0]) // 2
    start_y = (image_size[1] - rect_size[1]) // 2

    # Set the rectangle area to 1
    array[start_y:start_y + rect_size[1], start_x:start_x + rect_size[0]] = 1

    # Convert binary to 0 and 255 for BMP
    array *= 255
    img = Image.fromarray(array, mode="L")
    img.save("centered_rectangle.bmp", format="BMP")
    print("Centered rectangle image saved as 'centered_rectangle.bmp'.")


create_centered_rectangle_bmp()