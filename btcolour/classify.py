import btcolour
import btcolour.fit
import numpy as np
import matplotlib.pyplot as plt

def classify():
    return None

def plot_colour_space(img_path):
    guesses = btcolour.fit.fit(img_path, buffer=8)
    rgbs = [[i[3],i[4],i[5]] for i in guesses]
    rgbs = norm_rgbs(rgbs)
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.scatter(rgbs[:,0],rgbs[:,1],rgbs[:,2],c=rgbs,s=200)
    
def norm_rgbs(rgbs):
    return np.array([[i[0]/np.sum(i), i[1]/np.sum(i), i[2]/np.sum(i)] for i in rgbs])

def colour_from_guess(guess):
    rgb = [[i[3],i[4],i[5]] for i in [guess]]
    #btcolour.fit.reconstruct()
    return norm_rgbs(rgb)[0]