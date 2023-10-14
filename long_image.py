from util import next_image
import numpy as np
from PIL import Image


# TODO: cmd line args, test
def long_image(pipe, image, prompt, nsteps, num_shifts=10, shiftx=128, shifty=0):
    """Produces a single long image by applying next_image num_shifts times."""
    arr = np.zeros((512 + shifty * num_shifts, 512 + shiftx * num_shifts, 3), dtype=np.uint8)
    arr[:512, :512] = image

    front = image

    for i in range(1, num_shifts + 1):
        print(f"{i}/{num_shifts}")
        front = next_image(pipe, front, prompt, nsteps, shiftx, shifty)
        arr[i * shifty: i * shifty + 512, i * shiftx: i * shiftx + 512] = np.array(front)

    return Image.fromarray(arr)
