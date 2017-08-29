'''
author: Mengde Xu
date  : 2017-8-29
'''
import numpy as np
from matplotlib import pyplot as plt
import time


def reverse(img, alpha=True):
    '''
    reverse a img(W,H,C) or (W,H)
    Args:
        -img: img need to reverse
        -alpha:whether reverse it's alpha channel if it exists.
    '''
    if img.ndim > 2:
        W, H, C = img.shape
        if C <= 3:
            img = 255 - img
        else:
            if alpha:
                img = 255 - img
            else:
                img[:, :, :3] = 255 - img[:, :, :3]
    else:
        img = 255 - img

    return img


def log_transformation(img):
    '''
     reverse a img(W,H,C) or (W,H)
    Args:
        -img: img need to reverse
    '''
    if img.ndim > 2:
        x = 255 * np.log(1 + 1.0 * img[:, :, :3]) / 8

        img[:, :, :3] = np.round(x) % 255
    else:
        img = np.log(1 + img)
    return img


def single_channel_count(channel_map):
    '''
    count single channel map
    Args:
        -channel_map size W,H
    Return:
        -count_map size L
    '''
    L = 256
    count_map = np.zeros(L)
    for i in range(L):
        count_map[i] = np.size(channel_map[channel_map == i])
    return count_map


def gray_count(img):
    '''
    counting the grey in a image.
    Args:
        -img size:W,H,C
    Return:
        -gray_map size:L
    '''
    L = 256
    W, H, C = img.shape
    gray_map = np.zeros((C, L))
    for i in range(C):
        gray_map[i] = single_channel_count(img[:, :, i])
    return gray_map


def hist_show(img):
    '''
    hist show
    '''
    attr = ['r', 'g', 'b', 'alpha']

    static_map = gray_count(img)
    for i in range(img.shape[2]):
        plt.subplot(2, 2, i + 1)
        plt.bar(np.arange(256), static_map[i])
        plt.title(attr[i])
    plt.show()


def hist_equalization(_img):
    '''
    histgram equalization

    '''
    attr = ['r', 'g', 'b', 'alpha']

    L = 256
    W, H, C = _img.shape
    static_map = gray_count(_img)
    static_map = 1.0 * static_map / (W * H)
    # compute the s
    s = np.vstack([(L - 1) * np.sum(static_map[:, 0:i], axis=1)
                   for i in range(1, L + 1)]).transpose((1, 0))
    s = np.round(s)
    # show s in bar
    ###############
    # for i in range(C):
    #    plt.subplot(2, 2, i + 1)
    #    plt.bar(np.arange(256), s[i])
    #    plt.title(attr[i])
    # plt.show()
    ##############
    new_img = np.zeros_like(_img)
    for i in range(W):
        for j in range(H):
            for k in range(C):
                new_img[i, j, k] = s[k, _img[i, j, k]]
    # hist_show(new_img)
    return new_img

if __name__ == '__main__':

    img = plt.imread('lena.jpg')
    print(img.shape)
    # img[:,:,3]=img[:,:,3]-i
    #new_img1 = reverse(img, False)
    new_img = hist_equalization(img)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img)
    plt.axis('off')
    plt.title('After histgram-equalization')
    plt.show()
