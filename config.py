import torch

N_IMAGES = 50_000 # total number of images to generate

IMG_W, IMG_H = 256, 64 # images width and height

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "

MIN_LEN = 2 # minimum text length

MAX_LEN = 20 # maximum text length

ALLOWED = set(ALPHABET) # set of valid characters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIFFICULTY_RATIO = {0.2, 0.6, 0.2} # 20% clean, 60% medium and 20% hard augmentation for images generation

