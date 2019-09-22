from .kernels import *
from .imgconv import imgconv

def identity(imgpath, gray=True, **argkw):
    return imgconv(imgpath, identity_kernel(**argkw), strides=1, mean=False, gray=gray)

def sharpen(imgpath, gray=True, **argkw):
    return imgconv(imgpath, sharpen_kernel(**argkw), strides=1, mean=False, gray=gray)

def blur(imgpath, gray=True, **argkw):
    return imgconv(imgpath, blur_kernel(**argkw), strides=1, mean=False, gray=gray)

def emboss(imgpath, gray=True, **argkw):
    return imgconv(imgpath, emboss_kernel(**argkw), strides=1, mean=False, gray=gray)

def outline(imgpath, gray=True, **argkw):
    return imgconv(imgpath, outline_kernel(**argkw), strides=1, mean=True, gray=gray)

def sobel(imgpath, gray=True, **argkw):
    return imgconv(imgpath, sobel_kernel(**argkw), strides=1, mean=False, gray=gray)
