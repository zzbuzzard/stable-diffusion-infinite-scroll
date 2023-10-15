import numpy as np
from PIL import Image
import torch
import argparse


def generate_image(pipe, prompt, base_size, nsteps):
    return pipe(prompt=prompt, num_inference_steps=nsteps, height=base_size, width=base_size).images[0]


def generate_image_with_inpainting_pipeline(pipe, prompt, base_size, pipe_args):
    """
    Inpainting models (e.g. runwayml/stable-diffusion-inpainting) do not support the normal pipeline, so generating
    a non-inpainted image requires the workaround of using the inpainting pipeline.
    """
    image = np.zeros((base_size, base_size, 3), dtype=np.uint8)
    image = Image.fromarray(image)
    mask = np.zeros((base_size, base_size, 3), dtype=np.uint8) + 255
    mask = Image.fromarray(mask)
    image = pipe(prompt=prompt,
                 image=image,
                 mask_image=mask,
                 **pipe_args).images[0]
    return image


def next_image(pipe, image, base_size, prompt, shiftx, shifty, pipe_args):
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
                 **pipe_args).images[0]
    return image


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", nargs="+", help="Prompt list to use for generation. Separate prompts using "
                                                   "the '|' character.")
    parser.add_argument("-np", "--negative-prompt", nargs="?",
                        default="split image, two images, watermark, text", help="Negative prompt (optional). Defaults "
                                "to \"split image, two images, watermark, text\".")
    parser.add_argument("-cfg", "--guidance_scale", type=float, default=7.5, help="Context-free guidance")
    parser.add_argument("-s", "--steps", type=int, default=30, help="Number of steps to use for SD gen.")
    parser.add_argument("-as", "--attn-slicing", action='store_true',
                        help="Whether to use attention slicing. Attention slicing reduces memory usage at the cost of "
                             "increased inference time (unless your memory was already maxed out, in which case it "
                             "might be faster!")
    parser.add_argument("-ii", "--init-image", nargs="?",
                        help="Path to the init image. Will be scaled to the required resolution. If left blank, "
                             "the init image is generated with SD.")
    parser.add_argument("-d", "--direction", default="H", choices=["H", "V"],
                        help="Horizontal (H) or vertical (V) scroll mode. Default H.")
    parser.add_argument("-sh", "--shift", default=256, type=int,
                        help="How much (in pixels) to shift an image before applying inpainting to fill the space. "
                             "Default 256.")
    # "stabilityai/stable-diffusion-2-inpainting" is another option
    parser.add_argument("-m", "--model", default="runwayml/stable-diffusion-inpainting",
                        help="Stable Diffusion model to use. Can specify a local path or a HuggingFace model.")
    parser.add_argument("-r", "--res", default=512, type=int,
                        help="Resolution to run SD at. 512 recommended. Consider also modifying 'shift' if you modify "
                             "this. If larger values cause your GPU VRAM to max out, try using attention slicing (-as).")
    return parser
