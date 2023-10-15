from PIL import Image, ImageTk
import numpy as np


class Slider:
    def __init__(self, canvas, start_image, base_size, screen_width, screen_height, mode='H'):
        """
        :param canvas: Tk canvas
        :param start_image: Init image (must have shape base_size x base_size x 3)
        :param base_size: The size of the generated images (which must be square)
        :param screen_width: Width of screen in pixels
        :param screen_height: Height of screen in pixels
        :param mode: 'H' for horizontal, 'V' for vertical. Diagonal movement not currently supported.
        """
        assert mode in ['H', 'V'], f"Mode must be either 'H' or 'V', but found '{mode}'."
        start_image = np.array(start_image)
        assert start_image.shape == (base_size, base_size, 3), \
            f"Start image shape was {start_image.shape}, expected {(base_size, base_size, 3)}"
        self.canvas = canvas
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.mode = mode
        self.base_size = base_size

        # Size of the slider is bigger than the screen width by this much
        size_offset = 1024

        if mode == 'H':
            self.width = int(screen_width / screen_height * base_size) + size_offset
            self.height = base_size
            self.display_multiplier = screen_height / base_size  # from image pix to display pix
        else:
            self.width = base_size
            self.height = int(screen_height / screen_width * base_size) + size_offset
            self.display_multiplier = screen_width / base_size  # from image pix to display pix
        self.display_width = int(self.width * self.display_multiplier) + 1
        self.display_height = int(self.height * self.display_multiplier) + 1

        self.img_np = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.img_np[-base_size:, -base_size:] = np.array(start_image)

        # Start 256 away from the end
        self.xoffset = screen_width - self.display_width - base_size // 2

        self.img = Image.fromarray(self.img_np).resize((self.display_width, self.display_height),
                                                       Image.Resampling.LANCZOS)
        self.obj = ImageTk.PhotoImage(self.img)
        self.id = canvas.create_image((self.xoffset, 0), image=self.obj)

        self.speed = 0

    def move(self, deltatime):
        self.xoffset -= deltatime * self.speed

        if self.xoffset > 0:
            print("Gap on left")
            self.xoffset = 0
            self.speed *= 1.25  # just until the next generation...
        if self.xoffset < self.screen_width - self.display_width:
            print("Gap on right")
            self.xoffset = self.screen_width - self.display_width
            self.speed /= 1.25  # just until the next generation...

        self.canvas.moveto(self.id, self.xoffset, 0)

    def update(self, newimage, shiftx, shifty, speed):
        # TODO: Support shifty
        newimage = np.array(newimage)
        if self.mode == 'H':
            self.img_np = np.concatenate((self.img_np[:, shiftx:], newimage[:, self.base_size - shiftx:]), axis=1)
            shift = shiftx
        elif self.mode == 'V':
            self.img_np = np.concatenate((self.img_np[shifty:], newimage[self.base_size - shifty:]), axis=0)
            shift = shifty

        # Delete old object
        self.canvas.delete(self.id)

        # Create new object
        self.img = Image.fromarray(self.img_np). \
            resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.obj = ImageTk.PhotoImage(self.img)

        self.xoffset += shiftx * self.display_multiplier

        self.id = self.canvas.create_image((self.xoffset, 0), image=self.obj)

        # Set speed
        pix_per_sec = shift / max(0.1, speed)
        display_pix_per_sec = pix_per_sec * self.screen_height / 512
        ema_factor = 0.8  # higher = adapts faster to changes in speed (but less smooth)
        if self.speed == 0:  # first run
            self.speed = display_pix_per_sec
        else:
            self.speed = display_pix_per_sec * ema_factor + self.speed * (1 - ema_factor)
