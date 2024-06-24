import numpy as np
import pickle as pk
import json
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from btcolour.classify import *
import os.path
from scipy.stats import multivariate_normal

def fit(file_path, initial=None, buffer=7, indexes=None, show=False, function="gauss"):
    """
    :param file_path: absolute path to the json file
    :param initial: initial guess for the fit function
    :param buffer: pixels in each direction around the tag location
    :param indexes: indexes of tags to look at, if not passes, all are processed
    :param show: produces images of the tags and the fitted function
    """
    img, tags = image_loader(file_path)
    ex_tags = get_tags(img,tags,buffer,indexes,show)

    guesses = []

    for tag in ex_tags:
        guess = fit_from_img(tag, initial, function=function)
        guesses.append(guess)
        if show == True:
            fig, axs = plt.subplots(2)
            axs[0].imshow(tag)
            axs[1].imshow(reconstruct(guess,50,buffer,function))
            axs[1].scatter(40,40,color=colour_from_guess(guess),s=600)
    
    return guesses

def fit_from_img(img, initial=None, function="gauss"):
    r,g,b = extract_channels(img)

    x_data, y_data = np.meshgrid(np.arange(np.shape(r)[1]),np.arange(np.shape(r)[0]))

    channel_data = np.stack([r.flatten(),g.flatten(),b.flatten()]).ravel()
    height_data = channel_data#.flatten()

    x_data -= np.shape(r)[1]//2
    y_data -= np.shape(r)[0]//2
    coords = np.c_[x_data.flatten(),y_data.flatten()].T

    if function == "gauss":
        initial_guess = np.array([0,0,1,50,50,50,50,50,50])
        func = gauss
        
    elif function == "2d":
        initial_guess = np.array([0,0,1,50,50,50,50,50,50,1,0])
        func = gauss_2d
        
    if initial != None:
        initial_guess = initial
       
    #initial_guess_2d = np.array([0,0,1,1,50,50,0,50,50,50,50])
    
    guess, guess_cov = curve_fit(func,coords,height_data,p0=initial_guess)
    
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
    Returns image matrix and tag positions array taken from file_path of a json
    Requires absolute path
    '''
    parents, img_name = os.path.split(file_path)
    img_name = img_name[:-4] + 'np'
    parents, camID = os.path.split(parents)
    parents, _ = os.path.split(parents)
    img_path = os.path.join(parents, camID)
    img_path = os.path.join(img_path,img_name)
    
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

def gauss_2d(coords, x0, y0, sx, A0, A1, A2, B0, B1, B2, sy, theta):
    """
    Doesn't work properly
    """
    x, y = coords
    xo = float(x0)
    yo = float(y0)   
    a = (np.cos(theta)**2)/(2*sx**2) + (np.sin(theta)**2)/(2*sy**2)
    b = -(np.sin(2*theta))/(4*sx**2) + (np.sin(2*theta))/(4*sy**2)
    c = (np.sin(theta)**2)/(2*sx**2) + (np.cos(theta)**2)/(2*sy**2)
    gauss = np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    r = (B0 + A0*gauss).ravel()
    g = (B1 + A1*gauss).ravel()
    b = (B2 + A2*gauss).ravel()
    return np.array([r,g,b]).ravel()
    
def disc_func(coords, x0, y0, R, A0, A1, A2, B0, B1, B2):
    x, y = coords
    
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    red = A0 * (r <= R) + B0
    green = A1 * (r <= R) + B1
    blue = A2 * (r <= R) + B2
    
    return np.array([red,green,blue]).ravel()

def reconsruct_gauss(coords, size, x0, y0, spread, A0, A1, A2, B0, B1, B2):
    x, y = coords
    x = x / size
    y = y / size
    gauss = np.exp(-((x-x0)**2 + (y - y0)**2)/((spread**2)*2))
    red = A0*gauss + B0
    green = A1*gauss + B1
    blue = A2*gauss + B2

    for col in [red,green,blue]:
        col[col>255] = 255

    return np.array([red,green,blue])

def reconstruct_2d(coords,size, x0, y0, sx, A0, A1, A2, B0, B1, B2, sy, theta):
    """
    Doesn't work properly
    """
    x, y = coords
    x = x / size
    y = y / size
    
    xo = float(x0)
    yo = float(y0)   
    a = (np.cos(theta)**2)/(2*sx**2) + (np.sin(theta)**2)/(2*sy**2)
    b = -(np.sin(2*theta))/(4*sx**2) + (np.sin(2*theta))/(4*sy**2)
    c = (np.sin(theta)**2)/(2*sx**2) + (np.cos(theta)**2)/(2*sy**2)
    gauss = np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    r = (B0 + A0*gauss)
    g = (B1 + A1*gauss)
    b = (B2 + A2*gauss)
    
    return np.array([r,g,b])

def reconstruct(guess, size, buffer_size, function):
    x,y = np.meshgrid(np.arange(size*2),np.arange(size*2))
    x -= size
    y -= size
    coords = np.c_[x.flatten(),y.flatten()].T
    scale = size / buffer_size
    if function == "gauss":
        rav_image = reconsruct_gauss(coords,scale,*guess).T
        rav_image = rav_image.reshape(size*2,size*2,3).astype(np.uint16)
    elif function == "2d":
        rav_image = reconstruct_2d(coords,scale,*guess).T
        rav_image = rav_image.reshape(size*2,size*2,3).astype(np.uint16)
        print(np.shape(rav_image))
    
    return rav_image
