'''
@author: Mendel
@date  : 2017-8-27

interpolation in Python.

Including:
Nearest interpolation.
'''
import numpy as np


def Nearest_Interpolation(img, W_new, H_new, dim=1):
    '''
    img : ndarray,(N,C,W,H) or (N,W,H),this should be declared in parameters dim.
    dim: int,dim=1 if C exists(usually it's 3 for rgb.) else dim should be set as 0 which means the picture only have one channel.
    '''
    N, C, W, H = None, None, None, None
    if dim == 1:
        N, C, W, H = img.shape
        new_img = np.zeros((N, C, W_new, H_new), dtype=np.uint8)
    else:
        N, W, H = img.shape
        new_img = np.zeros((N, W_new, H_new), dtype=np.uint8)
    X = np.arange(W_new, dtype=int)
    Y = np.arange(H_new, dtype=int)
    # every point in the array is its coordinate.

    w_ratio = W / W_new
    h_ratio = H / H_new

    X_origin = np.round(X * w_ratio).astype(int)
    Y_origin = np.round(Y * h_ratio).astype(int)

    X_origin, Y_origin = np.meshgrid(X_origin, Y_origin)
    if dim == 1:
        new_img = np.vstack(img[:, :, X_origin, Y_origin]).reshape(
            N, C, H_new, W_new).transpose((0, 1, 3, 2))
    else:
        new_img = np.vstack(img[:, :, X_origin, Y_origin]).reshape(
            N, H_new, W_new).transpose((0, 2, 1))

    return new_img


def DoubleLinear_Interpolation(img, T, W_new, H_new):
    '''
    new_img=img*T

    '''
    img_point = np.ones((W_new, H_new, 3))
    C, W, H = img.shape
    X = np.arange(W_new)
    Y = np.arange(H_new)
    X, Y = np.meshgrid(X, Y)
    img_point[:, :, 0] = X.transpose((1,0))
    img_point[:, :, 1] = Y.transpose((1,0))

    point_old = img_point.dot(np.linalg.inv(T))


    x_left = np.floor(point_old[:, :, 0]).astype(int)
    y_top = np.floor(point_old[:, :, 1]).astype(int)

    x_left[x_left<0]=0
    x_left[x_left>=W]=W-1
    y_top[y_top<0]=0
    y_top[y_top>H]=H-1

    x_right = x_left + 1
    y_bottom = y_top + 1
    x_right[x_right>=W]=W-1
    y_bottom[y_bottom>=H]=H-1
    #new_img = np.vstack(img[:, X_origin, Y_origin])
    img1=np.vstack(img[:,x_left,y_top]).reshape(C,W_new,H_new)
    img2=np.vstack(img[:,x_left,y_bottom]).reshape(C,W_new,H_new)
    img3=np.vstack(img[:,x_right,y_top]).reshape(C,W_new,H_new)
    img4=np.vstack(img[:,x_right,y_bottom]).reshape(C,W_new,H_new)
    img5=((point_old[:,:,0]-1.0*x_left)*(img3-img1)+img1)
    img6=((point_old[:,:,0]-1.0*x_left)*(img4-img2)+img2)
    new_img=(point_old[:,:,1]-1.0*y_top)*(img6-img5)+img5

    return new_img


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    img = plt.imread('lena.jpg')
    img = img[:, :, :].transpose((2, 0, 1))
    T = np.array([[1, 0, 0], [0,2, 0], [0, 0, 1]])
    new_img = DoubleLinear_Interpolation(
        img, T,512, 750).transpose((1, 2, 0))

    plt.imshow(new_img.astype(np.uint8))
    plt.show()
