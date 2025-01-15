import numpy as np
from PIL import Image
import sys

from functions.morph import dilation, erosion, opening, closing, hit_or_miss, create_B2_from_B1, m4_operation_hmt, region_growing, m1
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
image_path = "./task3/centered_rectangle.bmp"
# image_path = "./task3/images/dilation.bmp"
# image_path = "./task3/binary_image.bmp"
# image_path = "./task3/images/pentagon.bmp"
# image_path = "./task3/images/c_lenac_small.bmp"
# image_path = "./task3/images/lena.bmp"
# image_path = "./task3/images/mandrilc.bmp"
# image_path = "./task3/images/girlc.bmp"

image = Image.open(image_path).convert('L') # task 3
# image = Image.open(image_path) # task 4
arr = np.array(image)

if len(arr.shape) == 3 and arr.shape[2] == 3:
    print("Detected an RGB image.")
    is_color = True
else:
    print("Detected a grayscale (or single-channel) image.")
    is_color = False

if command != "--region" and command != "--region-multi":
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
    result_arr = m4_operation_hmt(arr, B1_list, max_iterations=100)

elif command == '--m1':

    m1_B = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=int)
    result_arr = m1(arr, m1_B)


elif command == '--region':
    threshold = 350
    seed = (256, 256)

    print("Image shape:", arr.shape)
    print("Seed is:", seed)

    result_arr = region_growing(arr, seed, threshold, is_color=is_color)

elif command == '--region-multi':
    seeds = [(256, 256), (50, 50), (450, 100)]
    threshold = 300

    # 2) Preparing a labeled_result array
    rows, cols = arr.shape[:2]
    labeled_result = np.zeros((rows, cols), dtype=np.uint8)

    # 3) For each seed, run region growing with its unique label
    for idx, seed in enumerate(seeds, start=1):
        partial_result = region_growing(
            arr,
            seed=seed,
            threshold=threshold,
            region_value=idx,    # label = 1,2,3
            is_color=is_color
        )

        # 3a) Save or visualize partial_result (for debug)
        # partial_result has value `idx` where region is grown, 0 otherwise
        partial_binary = (partial_result == idx).astype(np.uint8)

        # Convert to image and save
        partial_im = Image.fromarray((partial_binary * 255).astype(np.uint8))
        partial_im.save(f"task3/results/region_seed_{idx}.bmp")
        print(f"Intermediate result for seed #{idx} saved as region_seed_{idx}.bmp")

        # 3b) Combine into the global labeled_result
        labeled_result[partial_result == idx] = idx

    # 4) Now we have a single labeled_result with labels 1..N
    # Let's save or visualize it (in a quick grayscale manner)
    # We'll do a "false color" approach: multiply each label by 50 for demonstration
    # (so label=1 => 50, label=2 => 100, label=3 => 150, etc.)
    colored_labels = (labeled_result * 50).astype(np.uint8)
    color_im = Image.fromarray(colored_labels)
    color_im.save("task3/results/labeled_result_multi.bmp")
    print("Combined labeled result saved as 'labeled_result_multi.bmp'.")

    # 5) Merge those labeled regions if they're similar
    from functions.morph import merge_regions
    merged_result = merge_regions(labeled_result, threshold=threshold)

    merged_colored = (merged_result * 50).astype(np.uint8)
    merged_im = Image.fromarray(merged_colored)
    merged_im.save("task3/results/labeled_result_multi_merged.bmp")
    print("Merged labeled result saved as 'labeled_result_multi_merged.bmp'.")

    result_arr = merged_result

else:
    print(f"Unknown command: {command}")
    print_help()
    sys.exit()

# Save the result
new_im = Image.fromarray((result_arr * 255).astype(np.uint8))
new_im.save("task3/results/result.bmp")
print("Result saved as 'result.bmp'.")
