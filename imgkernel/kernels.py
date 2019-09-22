import numpy as np

# 鲜明
def identity_kernel(iden=1.0):
    return np.array([[0, 0,    0],
                     [0, iden, 0],
                     [0, 0,    0]])

# 锐化
def sharpen_kernel(inner=5.0,  edge=-1.0):
    return np.array([[0,    edge,  0],
                     [edge, inner, edge],
                     [0,    edge,  0]])

# 模糊
def blur_kernel(inner=0.25,  edge=0.125, corner=0.0625):
    return np.array([[corner, edge,  corner],
                     [edge,   inner, edge],
                     [corner, edge,  corner]])

# 浮雕
def emboss_kernel(diag=2.0, iden=1.0):
    return np.array([[-diag, -iden, 0],
                     [-iden, iden,  iden],
                     [0,     iden,  diag]])

# 轮廓线
def outline_kernel(inner=8.0, outer=-1.0):
    return np.array([[outer, outer, outer],
                     [outer, inner, outer],
                     [outer, outer, outer]])

# 边缘检测
def sobel_kernel(direction, base=None, edge=2.0, corner=1.0):
    if base is not None:
        edge = base
        corner = base / 2
    if direction == 'top':
        return np.array([[corner, edge, corner], [0, 0, 0], [-corner, -edge, -corner]])
    elif direction == 'bottom':
        return np.array([[-corner, -edge, -corner], [0, 0, 0], [corner, edge, corner]])
    elif direction == 'left':
        return np.array([[corner, 0, -corner], [edge, 0, -edge], [corner, 0, -corner]])
    elif direction == 'right':
        return np.array([[-corner, 0, corner], [-edge, 0, edge], [-corner, 0, corner]])

    return identity_kernel()
