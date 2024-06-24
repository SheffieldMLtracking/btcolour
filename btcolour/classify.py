import btcolour
import btcolour.fit
import numpy as np
import matplotlib.pyplot as plt
import os.path

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
    rgbs = np.array([[i[0]/np.sum(i), i[1]/np.sum(i), i[2]/np.sum(i)] for i in rgbs])
    rgbs[rgbs<0]=0
    return rgbs

def colour_from_guess(guess):
    rgb = [[abs(i[3]),abs(i[4]),abs(i[5])] for i in [guess]]
    #btcolour.fit.reconstruct()
    return np.abs(norm_rgbs(rgb)[0])

def to_json_list(img_path_list):
    jsons = []
    for img in img_path_list:
        parents, img_name = os.path.split(img)
        img_name = img_name[:-2]+"json"
        json_name = os.path.join(parents,"btviewer",img_name)
        jsons.append(json_name)
    return jsons

def plot_fitted_colour(image_list,function="2d"):
    json_list = to_json_list(image_list)
    count = 0
    fig, axs = plt.subplots(3,4)
    guesses = []
    for file in json_list:
        img_raw, tag_list = btcolour.fit.image_loader(file)
        tags = btcolour.fit.get_tags(img_raw,tag_list,buffer=7)

        for i, tag in enumerate(tags):
            found = False
            guess = None
            buffer = 6
            while not found and buffer < 20:
                try:
                    guess = btcolour.fit.fit_from_img(tag,function=function)
                    guesses.append(guess)
                    found = True
                except RuntimeError:
                    found = False
                    buffer += 1
                    tag = btcolour.fit.get_tags(img_raw,tag_list,buffer,indexes=[i])
            if not found:
                print("tag %s wasn't fitted in image %s", (i, os.path.split(file)[1]))

            elif count > 11:
                fig, axs = plt.subplots(3,4)
                count = 0

            else:
                try:
                    axs[count//4,count%4].imshow(tag)
                except TypeError:
                    print("BLACK AND WHITE: "+file)
                count += 1
                axs[count//4,count%4].imshow(btcolour.fit.reconstruct(guess,50,buffer,function))
                axs[count//4,count%4].scatter(40,40,c=[colour_from_guess(guess)],s=400)
                count+=1

        
    plot_guesses(guesses)
    return guesses

def plot_guesses(guesses):
    rgbs = [[i[3],i[4],i[5]] for i in guesses]
    rgbs = norm_rgbs(rgbs)
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    print(rgbs)
    ax.scatter(rgbs[:,0],rgbs[:,1],rgbs[:,2],c=rgbs,s=200)
 
