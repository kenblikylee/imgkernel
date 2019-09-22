from PIL import Image
import numpy as np

def read_img(imgpath, gray=True):
    img = Image.open(imgpath)
    if gray:
        img = img.convert("L")
    return img

def crop_center(img):
    w, h = img.size
    c_x, c_y = w/2, h/2
    offset = min(w, h) / 2
    crop_box = (c_x-offset, c_y-offset, c_x+offset, c_y+offset)
    return img.crop(crop_box)

def resize_width(img, width):
    w, h = img.size
    height = int(width * h / w)
    return img.resize((width, height))

def apply_img_kernel(img, kernel, strides=1, mean=False):
    img = np.asarray(img)

    h, w = img.shape[:2]
    k_h, k_w = kernel.shape[:2]
    x_range = range(0, w - k_w + 1, strides)
    y_range = range(0, h - k_h + 1, strides)

    if mean:
        prosum = lambda a,b: min((a*b).mean(), 255)
    else:
        prosum = lambda a,b: min((a*b).sum(), 255)

    cal = lambda img: np.array([[prosum(img[i:i+k_h, j:j+k_w], kernel)
                                 for j in x_range]
                                    for i in y_range]).astype(np.uint8)

    if len(img.shape) == 2:
        data = cal(img)
        return Image.fromarray(data)
    elif len(img.shape) == 3:
        r, g, b = np.transpose(img, (2, 0, 1))
        _r, _g, _b = cal(r), cal(g), cal(b)
        return Image.merge('RGB', [Image.fromarray(d) for d in [_r, _g, _b]])

imgs = dict()

def imgconv(imgpath, kernel, strides=1, mean=False, gray=True):
    if imgpath in imgs:
        img = imgs[imgpath]
    else:
        img = read_img(imgpath, gray)
        img = crop_center(resize_width(img, 320))
    return img, apply_img_kernel(img, kernel, mean=mean)
