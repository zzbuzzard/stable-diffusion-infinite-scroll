from PIL import Image, ImageTk
import numpy as np


class Slider:
    def __init__(self, canvas, start_image, screen_width, screen_height):
        self.canvas = canvas

        # Size of the slider is bigger than the screen width by this much
        size_offset = 1024

        self.width = int(screen_width / screen_height * 512) + size_offset
        self.height = 512
        self.display_multiplier = screen_height / 512  # from image pix to display pix
        self.display_width = int(self.width * self.display_multiplier) + 1
        self.display_height = int(self.height * self.display_multiplier) + 1

        self.img_np = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.img_np[-512:, -512:] = np.array(start_image)

        # Start 256 away from the end
        self.xoffset = screen_width - self.display_width - 256

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
        if self.xoffset < screen_width - self.display_width:
            print("Gap on right")
            self.xoffset = screen_width - self.display_width
            self.speed /= 1.25  # just until the next generation...

        self.canvas.moveto(self.id, self.xoffset, 0)

    def update(self, newimage, shiftamt, speed):
        # Construct new image array
        newimage = np.array(newimage)
        self.img_np = np.concatenate((self.img_np[:, shiftamt:], newimage[:, 512 - shiftamt:]), axis=1)

        # Delete old object
        self.canvas.delete(self.id)

        # Create new object
        self.img = Image.fromarray(self.img_np). \
            resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.obj = ImageTk.PhotoImage(self.img)

        self.xoffset += shiftamt * self.display_multiplier

        self.id = canvas.create_image((self.xoffset, 0), image=self.obj)

        # Set speed
        pix_per_sec = shift / max(0.1, speed)
        display_pix_per_sec = pix_per_sec * screen_height / 512
        ema_factor = 0.8  # higher = adapts faster to changes in speed (but less smooth)
        if self.speed == 0:  # first run
            self.speed = display_pix_per_sec
        else:
            self.speed = display_pix_per_sec * ema_factor + self.speed * (1 - ema_factor)
