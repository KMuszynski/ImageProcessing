from PIL import Image
import numpy as np

def create_binary_bmp():
    bitmap = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    array = np.array(bitmap, dtype=np.uint8) * 255

    img = Image.fromarray(array, mode="L")
    img.save("binary_image.bmp", format="BMP")
    print("Binary image saved as 'binary_image.bmp'.")

if __name__ == "__main__":
    create_binary_bmp()
