import numpy as np
from PIL import Image
import sys

from functions.morph import dilation, erosion, opening, closing, hit_or_miss, create_B2_from_B1, m4_operation_hmt, region_growing
from functions.utilities import to_binary


def print_help():
    help_text = """
        Usage: python3 program.py [command] [param]

        Commands:
            --help                     Show this help message

            --dilation                 Apply dilation
            --erosion                  Apply erosion
            --opening                  Apply opening
            --closing                  Apply closing
            --hmt                      Apply hit-or-miss transform
            --m4                       Apply M4 operation

        Examples:
            python3 program.py --dilation
            python3 program.py --erosion
            python3 program.py --opening
            python3 program.py --closing
            python3 program.py --hmt
            python3 program.py --m4
    """
    print(help_text)


if len(sys.argv) < 2:
    print("No command provided.\n")
    print_help()
    sys.exit()

command = sys.argv[1]

if command == '--help':
    print_help()
    sys.exit()

# image_path = "./task3/images/girl_small.bmp"
# image_path = "./task3/images/b_mandrill.bmp"
# image_path = "./task3/images/b_boatbw.bmp"
# image_path = "./task3/images/b_lenabw.bmp"
# image_path = "./task3/binary_image_from_array.bmp"
# image_path = "./task3/images/dilation.bmp"
# image_path = "./task3/binary_image.bmp"
# image_path = "./task3/images/pentagon.bmp"
image_path = "./task3/images/c_lenac_small.bmp"
# image = Image.open(image_path).convert('L')
image = Image.open(image_path)
arr = np.array(image)

if len(arr.shape) == 3 and arr.shape[2] == 3:
    print("Detected an RGB image.")
    is_color = True
else:
    print("Detected a grayscale (or single-channel) image.")
    is_color = False

if command != "--region":
    arr = to_binary(arr)

    binary_im = Image.fromarray((arr * 255).astype(np.uint8))
    binary_im.save("task3/results/binary.bmp")
    print("Binary image saved as 'binary.bmp'.")

    B =  np.array([
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=int)


if command == '--dilation':
    result_arr = dilation(arr, B)

elif command == '--erosion':
    result_arr = erosion(arr, B)

elif command == '--opening':
    result_arr = opening(arr, B)

elif command == '--closing':
    result_arr = closing(arr, B)

elif command == '--hmt':
    B1 = np.array([
        [2, 2, 0],
        [2, 1, 2],
        [2, 2, 2]
    ], dtype=int)

    # Complement B2
    B2 = create_B2_from_B1(B1)

    result_arr = hit_or_miss(arr, B1, B2)

elif command == '--m4':
    # Set 1
    B1_1 = np.array([
        [1, 2, 2],
        [1, 0, 2],
        [1, 2, 2]
    ], dtype=int)

    B1_2 = np.array([
        [1, 1, 1],
        [2, 0, 2],
        [2, 2, 2]
    ], dtype=int)

    B1_3 = np.array([
        [2, 2, 1],
        [2, 0, 1],
        [2, 2, 1]
    ], dtype=int)

    B1_4 = np.array([
        [2, 2, 2],
        [2, 0, 2],
        [1, 1, 1]
    ], dtype=int)

    # Set 2
    # B1_1 = np.array([
    #     [0, 0, 0],
    #     [2, 1, 2],
    #     [1, 1, 1]
    # ], dtype=int)
    #
    # B1_2 = np.array([
    #     [2, 0, 0],
    #     [1, 1, 0],
    #     [1, 1, 2]
    # ], dtype=int)
    #
    # B1_3 = np.array([
    #     [1, 2, 0],
    #     [1, 1, 0],
    #     [1, 2, 0]
    # ], dtype=int)
    #
    # B1_4 = np.array([
    #     [1, 1, 2],
    #     [1, 1, 0],
    #     [2, 0, 0]
    # ], dtype=int)

    B1_list = [
        B1_1,
        B1_2,
        B1_3,
        B1_4
    ]
    result_arr = m4_operation_hmt(arr, B1_list, max_iterations=5000)

elif command == '--region':
    threshold = 350
    seed = (30, 30)

    print("Image shape:", arr.shape)
    print("Seed is:", seed)

    result_arr = region_growing(arr, seed, threshold, is_color=is_color)

else:
    print(f"Unknown command: {command}")
    print_help()
    sys.exit()

# Save the result
new_im = Image.fromarray((result_arr * 255).astype(np.uint8))
new_im.save("task3/results/result.bmp")
print("Result saved as 'result.bmp'.")
