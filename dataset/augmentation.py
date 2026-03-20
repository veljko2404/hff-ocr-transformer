from PIL import ImageEnhance, Image, ImageFilter
import random
from config import *
import numpy as np

def contrast(img):
    c = random.uniform(0.7,1.3)
    return ImageEnhance.Contrast(img).enhance(c)

def gamma(img):
    g = random.uniform(0.75,1.25)
    arr = np.array(img)/255.0
    arr = np.power(arr, g)
    arr = (arr*255).astype(np.uint8)
    return Image.fromarray(arr)

def perspective(img):
    dx = random.uniform(-0.05,0.05) * IMG_W
    dy = random.uniform(-0.05,0.05) * IMG_H

    coeffs = (
        1, dx/IMG_W, 0,
        dy/IMG_H, 1, 0,
        random.uniform(-0.0005,0.0005),
        random.uniform(-0.0005,0.0005)
    )
    return img.transform((IMG_W,IMG_H), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

def gradient_bg():
    img = Image.new("L", (IMG_W, IMG_H))
    px = img.load()

    start = random.randint(200, 250)
    end = random.randint(200, 250)

    for x in range(IMG_W):
        val = int(start + (end - start) * x / IMG_W)
        for y in range(IMG_H):
            px[x, y] = val + random.randint(-5, 5)
    return img

def brightness(img):
    b = random.uniform(0.8, 1.2)
    return ImageEnhance.Brightness(img).enhance(b)

# Augmentation: salt-and-pepper noise (random pure black or white pixels)
def noise(img, a, b):
    img = img.copy()
    px = img.load()
    for _ in range(random.randint(a, b)):
        px[random.randint(0, IMG_W - 1), random.randint(0, IMG_H - 1)] = 0 if random.random() < 0.5 else 255
    return img

def gaussian_blur(img, amount):
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, amount)))

def rotate_img(img, degrees, bg):
        angle = random.uniform(-degrees, degrees)
        return img.rotate(angle, resample=Image.BILINEAR, fillcolor=bg)

FONTS = [
    "fonts/DejaVuSans.ttf",
    "fonts/DejaVu Sans Bold.ttf",
    "fonts/ARIAL.TTF",
    "fonts/Liberation Sans Regular.ttf",
    "fonts/BebasNeue-Regular.ttf",
    "fonts/InstrumentSerif-Regular.ttf",
    "fonts/Lexend-VariableFont_wght.ttf",
    "fonts/Montserrat-VariableFont_wght.ttf",
    "fonts/OpenSans-VariableFont_wdth,wght.ttf",
    "fonts/PlayfairDisplay-VariableFont_wght.ttf",
    "fonts/RobotoCondensed-VariableFont_wght.ttf",
    "fonts/RobotoMono-VariableFont_wght.ttf",
    "fonts/RobotoSlab-VariableFont_wght.ttf",
    "fonts/Roboto-VariableFont_wdth,wght.ttf",
]

