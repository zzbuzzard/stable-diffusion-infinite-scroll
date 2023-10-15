import numpy as np
from PIL import Image
import torch


def generate_image(pipe, prompt, base_size, nsteps):
    return pipe(prompt=prompt, num_inference_steps=nsteps, height=base_size, width=base_size).images[0]


def generate_image_with_inpainting_pipeline(pipe, prompt, base_size, nsteps):
    image = np.zeros((base_size, base_size, 3), dtype=np.uint8)
    image = Image.fromarray(image)
    mask = np.zeros((base_size, base_size, 3), dtype=np.uint8) + 255
    mask = Image.fromarray(mask)
    image = pipe(prompt=prompt,
                 image=image,
                 mask_image=mask,
                 num_inference_steps=nsteps,
                 height=base_size,
                 width=base_size).images[0]
    return image


def next_image(pipe, image, base_size, prompt, nsteps, shiftx, shifty):
    """Given an image, uses inpainting to produce the next image (which overlaps with the previous image)"""
    assert image.size == (base_size, base_size)

    image_n = np.array(image)
    image_n = np.concatenate((image_n[:, shiftx:],
                              np.zeros((base_size, shiftx, 3), dtype=np.uint8)),
                             axis=1)
    image_n = np.concatenate((image_n[shifty:],
                              np.zeros((shifty, base_size, 3), dtype=np.uint8)),
                             axis=0)
    image = Image.fromarray(image_n)

    mask = np.zeros((base_size, base_size, 3), dtype=np.uint8)
    mask[:, base_size - shiftx:] = 255
    mask[base_size - shifty:, :] = 255
    mask = Image.fromarray(mask)

    image = pipe(prompt=prompt,
                 image=image,
                 mask_image=mask,
                 num_inference_steps=nsteps,
                 negative_prompt="split image",
                 height=base_size,
                 width=base_size).images[0]
    return image
