import random
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

import util

parser = util.get_argparser()
parser.add_argument("-n", "--nshifts", type=int, default=10,
                    help="Number of shifts to apply when generating the long image")
parser.add_argument("-out", "--out", type=str, required=True, help="Location to save the resulting long image")


def long_image(pipe, image, prompts, base_size, num_shifts, shiftx, shifty, pipe_args):
    """Produces a single long image by applying next_image num_shifts times."""
    arr = np.zeros((base_size + shifty * num_shifts, base_size + shiftx * num_shifts, 3), dtype=np.uint8)
    arr[:base_size, :base_size] = image

    front = image

    for i in range(1, num_shifts + 1):
        prompt = random.choice(prompts)
        print(f"Applying shift {i}/{num_shifts}")
        front = util.next_image(pipe, front, base_size, prompt, shiftx, shifty, pipe_args)
        arr[i * shifty: i * shifty + base_size, i * shiftx: i * shiftx + base_size] = np.array(front)

    return Image.fromarray(arr)


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading SD...")
    pipe = util.get_pipe(args.model, args.attn_slicing, args.legacy, xl=args.xl)
    print("Loaded.")

    prompts = " ".join(args.prompts).split("|")
    print(f"{len(prompts)} PROMPTS:\n", "\n  ".join(prompts))

    # Extra args for diffusers pipes
    pipe_args = {
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": " ".join(args.negative_prompt),
        "height": args.res,
        "width": args.res
    }

    if args.init_image is not None:
        start_image = Image.open(args.init_image).convert("RGB")
        start_image = start_image.resize((args.res, args.res))
    else:
        prompt = random.choice(prompts)
        print(f"No init image provided: generating from prompt '{prompt}'")
        start_image = util.generate_image_with_inpainting_pipeline(pipe, prompt, args.res, pipe_args)

    if args.direction == "H":
        shiftx = args.shift
        shifty = 0
    else:
        shiftx = 0
        shifty = args.shift

    im = long_image(pipe, start_image, prompts, args.res, args.nshifts, shiftx, shifty, pipe_args)
    if not args.out.endswith(".png") and not args.out.endswith(".jpg"):
        args.out += ".png"
    im.save(args.out)
