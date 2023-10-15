import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import tkinter as tk
import time
from multiprocessing import Process, Queue
import random
import argparse

import util
from util import next_image
from slider import Slider

parser = util.get_argparser()
parser.add_argument("-spd", "--speed", default=1., type=float,
                    help="Speed multiplier (between 0 and 1). A value of 1 causes images to be generated as fast as "
                         "possible. A value less than 1 leads to intentional breaks between generations to stop your "
                         "GPU exploding")


def draw_loop(queue, shiftx, shifty):
    """Repeatedly tells the slider to move, and notifies it when new images become available."""
    queue.get()  # wait from signal from update_loop to start
    print("Starting draw")

    start = time.time()
    prev = start

    while True:
        if not queue.empty():
            image, speed = queue.get()
            slider.update(image, shiftx, shifty, speed)

        t = time.time()
        slider.move(t - prev)
        prev = t

        root.update()


def generate_loop(queue, start_image, prompts, pipe_args, shiftx, shifty, model, base_size, attn_slicing, speed_mul=1):
    """
    Repeatedly computes new images to display using SD, and adds them to the queue.
    If speed_mul < 1, we wait between generations to reduce GPU usage intensity.
    """
    assert 0 < speed_mul <= 1
    print("Loading SD...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe.safety_checker = None  # A single black image causes a lot of problems for this scroller
    pipe = pipe.to("cuda")
    if attn_slicing:
        pipe.enable_attention_slicing()
    print("Loaded.")

    if start_image is None:
        prompt = random.choice(prompts)
        print(f"No init image provided: generating from prompt '{prompt}'")
        start_image = util.generate_image_with_inpainting_pipeline(pipe, prompt, base_size, pipe_args)

    queue.put(0)  # draw_loops waits for this to signal it should begin

    front = start_image
    while True:
        prompt = random.choice(prompts)
        print(f"Using prompt '{prompt}'")
        start = time.time()
        front = next_image(pipe, image=front, base_size=base_size, prompt=prompt, shiftx=shiftx, shifty=shifty,
                           pipe_args=pipe_args)
        duration = time.time() - start
        if speed_mul < 1:
            time.sleep(duration / speed_mul - duration)
        queue.put((front, duration / speed_mul))


if __name__ == "__main__":
    args = parser.parse_args()

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
        start_image = None
    speed_mul = max(min(args.speed, 1), 0.01)

    # Load GUI window
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    canvas = tk.Canvas(root,
                       width=screen_width,
                       height=screen_height,
                       highlightthickness=0)
    canvas.pack()

    if args.direction == "H":
        shiftx = args.shift
        shifty = 0
    else:
        shiftx = 0
        shifty = args.shift

    slider = Slider(canvas, start_image, args.res, screen_width, screen_height, mode=args.direction)

    queue = Queue()
    update_process = Process(target=generate_loop,
                             args=(queue, start_image, prompts, pipe_args, shiftx, shifty, args.model, args.res,
                                   args.attn_slicing, speed_mul))

    root.bind("<Escape>", lambda x: (root.destroy(), update_process.kill(), quit(0)))

    print("Starting update thread...")
    update_process.start()

    draw_loop(queue, shiftx, shifty)
