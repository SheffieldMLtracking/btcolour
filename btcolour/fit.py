import numpy as np
import pickle as pk
import json
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit(file_path, initial=None, buffer=7, indexes=None, show=False):
    img, tags = image_loader(file_path)
    ex_tags = get_tags(img,tags,buffer,indexes,show)

    guesses = []

    for tag in ex_tags:
        guesses.append(fit_from_img(tag, initial))
    
    return guesses

def fit_from_img(img, initial=None):
    r,g,b = extract_channels(img)

    x_data, y_data = np.meshgrid(np.arange(np.shape(r)[1]),np.arange(np.shape(r)[0]))

    channel_data = np.stack([r.flatten(),g.flatten(),b.flatten()]).ravel()
    height_data = channel_data#.flatten()

    x_data -= np.shape(r)[1]//2
    y_data -= np.shape(r)[0]//2
    coords = np.c_[x_data.flatten(),y_data.flatten()].T

    if initial != None:
        initial_guess = initial
    else:
        initial_guess = np.array([0,0,1,50,50,50,50,50,50])
    #initial_guess_2d = np.array([0,0,1,1,50,50,0,50,50,50,50])
    
    guess, guess_cov = curve_fit(gauss,coords,height_data,p0=initial_guess)
    
    return guess

def debay(img_mat):
    '''
    Debays the image, assumes BG -> RGB
    '''
    return np.array(cv2.cvtColor(img_mat, cv2.COLOR_BAYER_BG2RGB))

def extract_channels(img):
    '''
    Takes an image and extracts the r,g,b matrices from it
    '''
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    return r, g, b

def image_loader(file_path):
    '''
    Returns image matrix and tag positions array taken from file_path
    Requires absolute path
    '''
    img_path = file_path.split("/")
    img_path.pop(-3)
    img_path = "".join([i+"\\" for i in img_path])[:-5] + 'np'
    
    with open(img_path,'rb') as pick:
        image_arr = np.array(pk.load(pick)['img'])
        
    with open(file_path,'r') as tgs:
        tag_json = json.load(tgs)
    tags = []
    
    for tag in tag_json:
        coords = (tag['x'],tag['y'])
        tags.append(coords)    

    return debay(image_arr), tags

def get_tags(img, tags, buffer, indexes=None, show=False):
    '''
    Extracts the tags from an image, centered at the tag coordinates
    With buffer number of pixels surrounding in each direction
    '''
    extracted_tags = []
    if indexes != None: tags = [tags[i] for i in indexes]
    for i, t in enumerate(tags):
        x = t[0]
        y = t[1]
        tag = img[y-buffer:y+buffer,x-buffer:x+buffer]
        if show:
            plt.figure(figsize=(8,8))
            plt.imshow(tag)
            plt.title(i)
        extracted_tags.append(tag)
        
    return np.array(extracted_tags)

def gauss(coords, x0, y0, spread, A0, A1, A2, B0, B1, B2):
    """
    Assumes the variance is the same in the x and y
    :param coords: The x variable of the function
    :param x0: meanx of the normal distribution
    :param y0: meany of the n.d.
    :param spread: std. dev
    :param A0: amplitude of the red channel
    :param A1: amplitude of the green channel
    :param A2: amplitude of the blue channel
    :param B0: amplitude of the red channel background
    :param B1: amplitude of the green channel background
    :param B2: amplitude of the blue channel background

    """
    x, y = coords
    gauss = np.exp(-((x-x0)**2 + (y - y0)**2)/((spread**2)*2))
    red = A0*gauss + B0
    green = A1*gauss + B1
    blue = A2*gauss + B2

    return np.array([red,green,blue]).ravel()

def gauss_2d(coords, x0, y0, sx, sy, theta, A0, A1, A2, B0, B1, B2):
    """
    Doesn't work properly
    """
    x, y = coords
    a=np.cos(theta)*x -np.sin(theta)*y
    b=np.sin(theta)*x +np.cos(theta)*y
    a0=np.cos(theta)*x0 -np.sin(theta)*y0
    b0=np.sin(theta)*x0 +np.cos(theta)*y0
    
    gauss = np.exp(-(((a-a0)**2)/(2*(sx**2)) + ((b-b0)**2) /(2*(sy**2))))
    
    red = A0*gauss + B0
    green = A1*gauss + B1
    blue = A2*gauss + B2
    
    return np.array([red,green,blue]).ravel()

def disc_func(coords, x0, y0, R, A0, A1, A2, B0, B1, B2):
    x, y = coords
    
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    red = A0 * (r <= R) + B0
    green = A1 * (r <= R) + B1
    blue = A2 * (r <= R) + B2
    
    return np.array([red,green,blue]).ravel()
