import matplotlib.pyplot as plt
import matplotlib.image as img
import time

import numpy as np


def scaling(file,type=None):
    r,g,b = file[:,:,0],file[:,:,1],file[:,:,2]

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    z = [[i]*3 for i in gray]
    z= np.array(z,dtype=np.int64)
    z = np.transpose(z, (0,2,1))

    if type == "r":
        z[:,:,1] = 0
        z[:,:,2] = 0
        return z
    if type == "g":
        z[:,:,0] = 0
        z[:,:,2] = 0
        return z
    if type == "b":
        z[:,:,0] = 0
        z[:,:,1] = 0
        return z
    return z

if __name__ == "__main__":
    #testing
    file = "13189.jpg"
    print(f"getting image")
    start_time = time.time()
    m = img.imread(file)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(2, 3, 1)
    print(f"Image loaded, took {time.time()- start_time}.")
    plt.imshow(m)
    plt.axis('off')
    plt.title("Original")

    # Adds a subplot at the 2nd position
    fig.add_subplot(2, 3, 2)
    print(f"red scale the image")
    output = scaling(m,"r")
    plt.title("Red")

    plt.imshow(output)

    fig.add_subplot(2, 3, 3)
    start_time = time.time()
    print(f"red scale the image")
    output = scaling(m,"g")
    plt.title("Green")

    plt.imshow(output)

    fig.add_subplot(2, 3, 4)
    start_time = time.time()
    print(f"red scale the image")
    output = scaling(m,"b")
    plt.title("Blue")

    plt.imshow(output)

    fig.add_subplot(2, 3, 5)
    start_time = time.time()
    print(f"red scale the image")
    output = scaling(m)
    plt.title("Gray")
    plt.imshow(output)

    plt.show()