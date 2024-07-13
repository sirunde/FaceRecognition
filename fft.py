import base64
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import threading
import time
import cv2


def gkern(l=15, sig=3):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-int(l - 1) / 2., int(l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig)) * (1 / np.square(2 * np.pi) * (sig ** 2))
    kernel = np.outer(gauss, gauss)

    return kernel / np.sum(kernel)


def convolution(img, kernel):
    img_w, img_h = img.shape[:2]
    kernel_w, kernel_h = kernel.shape[:2]

    # rgb
    if (len(img.shape) == 3):
        image_pad = np.pad(img, pad_width=( \
            (kernel_h // 2, kernel_h // 2), (kernel_w // 2, \
                                             kernel_w // 2), (0, 0)), mode='constant', \
                           constant_values=0).astype(np.float32)

    # gray
    if (len(img.shape) == 2):
        image_pad = np.pad(img, pad_width=( \
            (kernel_h // 2, kernel_h // 2), (kernel_w // 2, \
                                             kernel_w // 2)), mode='constant', \
                           constant_values=0).astype(np.float32)

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image_pad.shape)

    for i in range(h, image_pad.shape[0] - h):
        for j in range(w, image_pad.shape[1] - w):
            # sum = 0
            if len(image_pad.shape) == 3:
                for lo in range(3):
                    x = image_pad[i - h:i + h + 1, j - w:j + w + 1, lo]
                    x = x * kernel
                    image_conv[i, j, lo] = x.sum()

    h_end = -h
    w_end = -w

    if (h == 0):
        return image_conv[h:, w:w_end]
    if (w == 0):
        return image_conv[h:h_end, w:]

    return image_conv[h:h_end, w:w_end, :]


def naive(file):
    print(f"getting image")
    start_time = time.time()
    m = img.imread(file)
    print(f"Image loaded, took {time.time() - start_time}.")

    print(m.shape)
    w, h = m.shape[:2]
    print(f"starting kernel process")
    start_time = time.time()
    kernel = gkern()
    print(f"kernel ended. Took {time.time() - start_time}")

    output = np.zeros(m.shape)

    print(f"starting convolution")
    start_time = time.time()
    out = (convolution(m, kernel))
    print(f"convolution ended. Took {time.time() - start_time}")

    plt.imshow(out.astype(np.uint8))
    plt.show()
    plt.imsave("output4.jpg", out.astype(np.uint8))


# creates fft
def fft(file, l=15, s=3):
    try:
        if (isinstance(file, (np.ndarray))):
            m = file
        if (isinstance(file, str)):
            try:
                m = img.imread(file)
            except Exception as e:
                print(e)
                exit(1)
        w, h = m.shape[:2]
        kernel = gkern(l, s)
        padded_kernel = np.pad(kernel, [(int((w - 15)/2,)+1, int((w - 15)/2,)), (int((h - 15)/2)+1,int((h - 15)/2))], 'constant', constant_values= 0)
        padded_kernel = np.fft.ifftshift(padded_kernel)
        ffted_kernel = (np.fft.fft2((padded_kernel)))
        ffted_file_list = ([(np.fft.fft2(m[:,:,i])) for i in range(m.shape[2])])
        matrix_multied = [i*ffted_kernel for i in ffted_file_list]
        output = np.transpose(np.array([np.fft.ifft2(i) for i in matrix_multied]),(1,2,0))
        plt.imshow(output.astype(np.uint8))
        plt.show()
        plt.imsave("output3.jpg", output.astype(np.uint8))
    except Exception as e:
        raise


if __name__ == "__main__":
    fileToRun = "13189.jpg"
    # naive(file)
    a = fft(fileToRun)
    # a = np
