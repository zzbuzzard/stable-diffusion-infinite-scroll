import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import time
from multiprocessing import Process, Queue
import random
import argparse  # TODO: Use

from util import next_image
from slider import Slider

# TODO:
#  - Prompt text files so u dont have to type every time
#  - Prompt randomisation / generate random CIP embeds?
#     - text file --> config file specifying random words to be used for prompts?
#  - Computer gets super hot: add some option to slow down? (breaks between gens lol)
#     - e.g. "speed multiplier" -> 1x is work constantly,
#                                0.5x is do a gen (20sec) then wait (20sec)

# TODO: Cmd line args
_shiftx = 256
_shifty = 0
sd_model = "runwayml/stable-diffusion-inpainting"
base_size = 512


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


def update_loop(queue, start_image, prompts, numsteps, shiftx, shifty, speed_mul=1):
    """
    Repeatedly computes new images to display using SD, and adds them to the queue.
    If speed_mul < 1, we wait between generations to reduce GPU usage intensity.
    """
    assert 0 < speed_mul <= 1
    print("Loading SD...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sd_model,
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
    path = input("Enter start image path: ")
    numsteps = int(input("How many steps? (10-50) "))
    speed_mul = float(input("Speed multiplier: (0-1) "))
    speed_mul = max(min(speed_mul, 1), 0.01)
    numprompts = int(input("How many prompts? "))
    prompts = [input(f"Prompt {i}: ") for i in range(1, numprompts + 1)]
    print("Ok... GPU time\n\n")

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

    start_image = Image.open(path).convert("RGB")
    start_image = start_image.resize((base_size, base_size))
    slider = Slider(canvas, start_image, base_size, screen_width, screen_height, mode='H')

    queue = Queue()
    update_process = Process(target=update_loop,
                             args=(queue, start_image, prompts, numsteps, _shiftx, _shifty, speed_mul))
    print("Starting update thread...")
    update_process.start()

    draw_loop(queue, _shiftx, _shifty)

    # res = long_image(image, prompt, num_shifts = 16, shiftx = 128, shifty = 0)
    # res.save(f"images/longfantasy.png")
