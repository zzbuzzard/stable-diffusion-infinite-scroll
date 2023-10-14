import numpy as np
from PIL import Image


def next_image(pipe, image, prompt, nsteps, shiftx, shifty):
    """Given an image, uses inpainting to produce the next image (which overlaps with the previous image)"""
    assert image.size == (512, 512)

    image_n = np.array(image)
    image_n = np.concatenate((image_n[:, shiftx:],
                              np.zeros((512, shiftx, 3), dtype=np.uint8)),
                             axis=1)
    image_n = np.concatenate((image_n[shifty:],
                              np.zeros((shifty, 512, 3), dtype=np.uint8)),
                             axis=0)
    image = Image.fromarray(image_n)

    mask = np.zeros((512, 512, 3), dtype=np.uint8)
    mask[:, 512 - shiftx:] = 255
    mask[512 - shifty:, :] = 255
    mask = Image.fromarray(mask)

    image = pipe(prompt=prompt,
                 image=image,
                 mask_image=mask,
                 num_inference_steps=nsteps,
                 negative_prompt="split image").images[0]
    return image
