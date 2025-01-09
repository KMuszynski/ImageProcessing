import numpy as np
from PIL import Image
import sys

from functions.morph import dilation, erosion, opening, closing, hit_or_miss
from functions.utilities import to_binary


def print_help():
    help_text = """
        Usage: python3 program.py [command] [param]

        Commands:
            --help                     Show this help message

            --dilation <param>         Apply dilation with a square SE of size <param>
            --erosion <param>          Apply erosion with a square SE of size <param>
            --opening <param>          Apply opening with a square SE of size <param>
            --closing <param>          Apply closing with a square SE of size <param>
            --hmt <param>              Apply hit-or-miss transform with a SE of size <param>
            --m4                       Apply M4 operation as described (no additional param)

        Examples:
            python3 program.py --dilation 3
            python3 program.py --erosion 3
            python3 program.py --opening 3
            python3 program.py --closing 3
            python3 program.py --hmt 3
            python3 program.py --m4
    """
    print(help_text)


# Parse arguments
if len(sys.argv) < 2:
    print("No command provided.\n")
    print_help()
    sys.exit()

command = sys.argv[1]

# Load the image
# image_path = "./task3/images/b_mandrill.bmp"
image_path = "./task3/binary_image.bmp"
image = Image.open(image_path).convert("L")
arr = np.array(image)
arr = to_binary(arr)

binary_im = Image.fromarray((arr * 255).astype(np.uint8))
binary_im.save("task3/results/binary.bmp")
print("Binary image saved as 'binary.bmp'.")


def create_B2_from_B1(B1):
    """
    Convert B1 to B2:
    - B1=1 => B2=0
    - B1=0 => B2=1
    - B1=2 => B2=2 (inactive stays inactive)
    """
    B2 = B1.copy()
    B2[B1 == 1] = 0
    B2[B1 == 0] = 1
    # B1==2 remain 2 (inactive)
    return B2

def m4_operation_hmt(A, B1_list, max_iterations=1000):
    """
    For each 3-value structuring element B1_i in B1_list:
      - Create B2_i from B1_i
      - Let X_{i,0} = A
      - Repeat:
           X_{i,k} = hit_or_miss(X_{i,k-1}, B1_i, B2_i)  UNION A
        until X_{i,k} == X_{i,k-1} or max_iterations
      - Let D_i = stable X_{i,k}
    Then H(A) = D_1 ∪ D_2 ∪ ... ∪ D_n
    """
    H = np.zeros_like(A, dtype=np.uint8)

    for idx, B1_i in enumerate(B1_list, start=1):
        # Build B2_i
        B2_i = create_B2_from_B1(B1_i)

        X_old = A.copy()
        iteration_count = 0

        while True:
            # Perform hit-or-miss on X_old
            hmt_res = hit_or_miss(X_old, B1_i, B2_i)

            # Union with A
            X_new = np.logical_or(hmt_res, A).astype(np.uint8)

            print(f"[M4-HMT] SE {idx}, Iteration {iteration_count}, Foreground:", np.sum(X_new))

            # Check convergence
            if np.array_equal(X_new, X_old):
                break

            X_old = X_new
            iteration_count += 1
            if iteration_count >= max_iterations:
                print("Warning: Reached max_iterations without convergence.")
                break

        # Union stable result with H
        H = np.logical_or(H, X_old).astype(np.uint8)

    return H


if command == '--help':
    print_help()
    sys.exit()

elif command in ['--dilation', '--erosion', '--opening', '--closing', '--hmt']:
    if len(sys.argv) < 3 and command not in ['--dilation', '--erosion', '--opening', '--closing']:
        print("Please provide the parameter for the selected operation.")
        print_help()
        sys.exit()
    if command == '--hmt':
        param = sys.argv[2]
        size = int(param)

    B =  np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0]
    ], dtype=int)   # np.ones((size, size), dtype=int)

    if command == '--dilation':
        result_arr = dilation(arr, B)
    elif command == '--erosion':
        result_arr = erosion(arr, B)
    elif command == '--opening':
        result_arr = opening(arr, B)
    elif command == '--closing':
        result_arr = closing(arr, B)
    elif command == '--hmt':
        if size < 3:
            print("HMT requires at least a 3x3 structuring element.")
            sys.exit()

        # Example B1 (cross)
        B1 = np.zeros((size, size), dtype=int)
        B1[size // 2, :] = 1
        B1[:, size // 2] = 1

        # Complement B2
        B2 = create_B2_from_B1(B1)

        result_arr = hit_or_miss(arr, B1, B2)

elif command == '--m4':
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

    B1_list = [
        B1_1,
        B1_2,
        B1_3,
        B1_4
    ]
    result_arr = m4_operation_hmt(arr, B1_list, max_iterations=1000)

else:
    print(f"Unknown command: {command}")
    print_help()
    sys.exit()

# Save the result
new_im = Image.fromarray((result_arr * 255).astype(np.uint8))
new_im.save("task3/results/result.bmp")
print("Result saved as 'result.bmp'.")
