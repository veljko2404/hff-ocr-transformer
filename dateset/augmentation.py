from PIL import ImageEnhance, Image, ImageFilter
import random
import numpy as np
from config import *


def contrast(img):
    c = random.uniform(0.5,1.7)
    return ImageEnhance.Contrast(img).enhance(c)

def gamma(img):
    g = random.uniform(0.5,1.5)
    arr = np.array(img)/255.0
    arr = np.power(arr, g)
    arr = (arr*255).astype(np.uint8)
    return Image.fromarray(arr)

def motion_blur(img):
    size = random.choice([3,5,7])
    kernel = np.zeros((size,size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return img.filter(ImageFilter.Kernel((size,size), kernel.flatten(), scale=1))

def perspective(img):
    dx = random.uniform(-0.15,0.15) * IMG_W
    dy = random.uniform(-0.15,0.15) * IMG_H

    coeffs = (
        1, dx/IMG_W, 0,
        dy/IMG_H, 1, 0,
        random.uniform(-0.0005,0.0005),
        random.uniform(-0.0005,0.0005)
    )
    return img.transform((IMG_W,IMG_H), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

def texture_bg():
    arr = np.random.normal(220, 15, (IMG_H, IMG_W))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")

def gradient_bg():
    img = Image.new("L", (IMG_W, IMG_H))
    px = img.load()

    start = random.randint(180, 255)
    end = random.randint(180, 255)

    for x in range(IMG_W):
        val = int(start + (end - start) * x / IMG_W)
        for y in range(IMG_H):
            px[x, y] = val + random.randint(-5, 5)
    return img

FONTS = [
    "fonts/DejaVuSans.ttf",
    "fonts/DejaVu Sans Bold.ttf",
    "fonts/ARIAL.TTF",
    "fonts/Liberation Sans Regular.ttf",
    "fonts/AlfaSlabOne-Regular",
    "fonts/BebasNeue-Regular",
    "fonts/Bungee-Regular",
    "fonts/DancingScript-VariableFont_wght",
    "fonts/InstrumentSerif-Regular",
    "fonts/Lexend-VariableFont_wght",
    "fonts/LobsterTwo-Regular",
    "fonts/Montserrat-VariableFont_wght",
    "fonts/OpenSans-VariableFont_wdth,wght",
    "fonts/Pacifico-Regular",
    "fonts/PlayfairDisplay-VariableFont_wght",
    "fonts/RobotoCondensed-VariableFont_wght",
    "fonts/RobotoMono-VariableFont_wght",
    "fonts/RobotoSlab-VariableFont_wght",
    "fonts/Roboto-VariableFont_wdth,wght",
]

