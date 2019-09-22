# imgkernel
An image kernel is a small matrix used to apply effects like the ones you might find in Photoshop or Gimp, such as blurring, sharpening, outlining or embossing. They're also used in machine learning for 'feature extraction', a technique for determining the most important portions of an image. 

## 安装

``` sh
pip install imgkernel
```

## 使用


```python
import imgkernel
import matplotlib.pyplot as plt
%matplotlib inline

def show2imgs(img1, img2, title=None):
    fig = plt.figure(figsize=(10, 5))

    if title is not None:
        fig.suptitle(title)

    plt.subplot(121)
    plt.axis('off')
    plt.imshow(img1)

    plt.subplot(122)
    plt.axis('off')
    plt.imshow(img2)

    plt.show()

imgpath = 'image.jpeg'
```

### 1. 鲜明


```python
show2imgs(*imgkernel.identity(imgpath, gray=False, iden=1.6), 'imgkernel.identity()')
```

![](http://cdn.kenblog.top/imgkernel_identity.png)

### 2. 模糊


```python
show2imgs(*imgkernel.blur(imgpath, gray=False), 'imgkernel.blur()')
```

![](http://cdn.kenblog.top/imgkernel_blur.png)

### 3. 锐利


```python
show2imgs(*imgkernel.sharpen(imgpath, gray=False, inner=1.7,  edge=-0.08), 'imgkernel.sharpen()')
```

![](http://cdn.kenblog.top/imgkernel_sharpen.png)

### 4. 浮雕


```python
show2imgs(*imgkernel.emboss(imgpath, gray=False), 'imgkernel.emboss()')
```

![](http://cdn.kenblog.top/imgkernel_emboss.png)

### 5. 轮廓线


```python
show2imgs(*imgkernel.outline(imgpath, gray=False, inner=8.9, outer=-1.29), 'imgkernel.outline()')
```

![](http://cdn.kenblog.top/imgkernel_outline.png)

### 6. 边沿检测

#### 6.1 上边沿


```python
show2imgs(*imgkernel.sobel(imgpath, gray=False, direction='top', base=0.03), 'imgkernel.sobel(top)')
```

![](http://cdn.kenblog.top/imgkernel_sobel_top.png)

#### 6.2 下边沿


```python
show2imgs(*imgkernel.sobel(imgpath, gray=False, direction='bottom', base=0.03), 'imgkernel.sobel(bottom)')
```

![](http://cdn.kenblog.top/imgkernel_sobel_bottom.png)

#### 6.3 左边沿


```python
show2imgs(*imgkernel.sobel(imgpath, gray=False, direction='left', base=0.03), 'imgkernel.sobel(left)')
```

![](http://cdn.kenblog.top/imgkernel_sobel_left.png)

#### 6.4 右边沿


```python
show2imgs(*imgkernel.sobel(imgpath, gray=False, direction='right', base=0.03), 'imgkernel.sobel(right)')
```

![](http://cdn.kenblog.top/imgkernel_sobel_right.png)
