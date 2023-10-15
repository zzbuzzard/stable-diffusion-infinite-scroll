import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import time
from multiprocessing import Process, Queue
import random
import argparse  # TODO: Use
import gc

import util
from util import next_image
from slider import Slider

parser = argparse.ArgumentParser()
parser.add_argument("prompts", nargs="+", help="Prompt list to use for generation.")
parser.add_argument("-s", "--steps", type=int, default=30, help="Number of steps to use for SD gen.")
parser.add_argument("-spd", "--speed", default=1., type=float,
                    help="Speed multiplier (between 0 and 1). A value of 1 causes images to be generated as fast as "
                         "possible. A value less than 1 leads to intentional breaks between generations to stop your "
                         "GPU exploding")
parser.add_argument("-ii", "--init-image", nargs="?",
                    help="Path to the init image. Will be scaled to the required resolution. If left blank, the init "
                         "image is generated with SD.")
parser.add_argument("-d", "--direction", default="H", choices=["H", "V"],
                    help="Horizontal (H) or vertical (V) scroll mode.")
parser.add_argument("-sh", "--shift", default=256, type=int,
                    help="How much (in pixels) to shift an image before applying inpainting to fill the space.")
# "runwayml/stable-diffusion-inpainting" is another good option
# "stabilityai/stable-diffusion-2-inpainting"
parser.add_argument("-m", "--model", default="runwayml/stable-diffusion-inpainting",
                    help="Stable Diffusion model to use. Can specify a local path or a HuggingFace model.")
# parser.add_argument("-mi", "--model-init", default="CompVis/stable-diffusion-v1-4",
#                     help="Stable Diffusion model to use for the *init image* only.")
parser.add_argument("-r", "--res", default=512, type=int,
                    help="Resolution to run SD at. 512 recommended; higher values will likely be much slower.")


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


def change_speed(x):
    slider.speed += x
    print("Speed =", slider.speed)


def update_loop(queue, start_image, prompts, numsteps, shiftx, shifty, model, base_size, speed_mul=1):
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
    print("Loaded.")

    queue.put(0)  # draw_loops waits for this to signal it should begin

    front = start_image
    while True:
        prompt = random.choice(prompts)
        print("Using prompt ' " + prompt + " '")
        start = time.time()
        front = next_image(pipe, image=front, base_size=base_size, prompt=prompt, nsteps=numsteps,
                           shiftx=shiftx, shifty=shifty)
        duration = time.time() - start
        if speed_mul < 1:
            time.sleep(duration / speed_mul - duration)
        queue.put((front, duration / speed_mul))


if __name__ == "__main__":
    args = parser.parse_args()

    prompts = args.prompts
    print("PROPTS:", prompts)

    if args.init_image is None:
        prompt = random.choice(prompts)
        print(f"No init image provided: generating from prompt '{prompt}'")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            args.model,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe.safety_checker = None
        pipe = pipe.to("cuda")

        start_image = util.generate_image_with_inpainting_pipeline(pipe, prompt, args.res, args.steps)
        # start_image = (start_image / 255).cpu().numpy().astype(np.uint8)
        # start_image = Image.fromarray(start_image)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        start_image = Image.open(args.init_image).convert("RGB")
        start_image = start_image.resize((args.res, args.res))

    # path = input("Enter start image path: ")
    # numsteps = int(input("How many steps? (10-50) "))
    # speed_mul = float(input("Speed multiplier: (0-1) "))
    speed_mul = max(min(args.speed, 1), 0.01)
    # numprompts = int(input("How many prompts? "))
    # prompts = [input(f"Prompt {i}: ") for i in range(1, numprompts + 1)]
    # print("Ok... GPU time\n\n")

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

    ##    spdup = tk.Button(root,
    ##                       text="spd up",
    ##                       command = lambda:changespeed(1),
    ##                       width = 10)
    ##    spddown = tk.Button(root,
    ##                       text="spd down",
    ##                       command = lambda:changespeed(-1),
    ##                       width = 10)
    ##    canvas.create_window(20,30,window=spdup,anchor='sw')
    ##    canvas.create_window(20,60,window=spddown,anchor='sw')

    if args.direction == "H":
        shiftx = args.shift
        shifty = 0
    else:
        shiftx = 0
        shifty = args.shift

    slider = Slider(canvas, start_image, args.res, screen_width, screen_height, mode=args.direction)

    queue = Queue()
    update_process = Process(target=update_loop,
                             args=(queue, start_image, prompts, args.steps, shiftx, shifty, args.model, args.res, speed_mul))

    root.bind("<Escape>", lambda x: (root.destroy(), update_process.kill()))

    print("Starting update thread...")
    update_process.start()

    draw_loop(queue, shiftx, shifty)

    # res = long_image(image, prompt, num_shifts = 16, shiftx = 128, shifty = 0)
    # res.save(f"images/longfantasy.png")
