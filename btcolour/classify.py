import btcolour
import btcolour.fit
import numpy as np
import matplotlib.pyplot as plt
import os.path
from sklearn.neighbors import KNeighborsClassifier

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

def train(rgbs, labels, k=3):
    """
    Takes in an array of rgb values as the x_data and array of corressponding tag ids(labels) as y_data
    Returns a K-nn classifier
    """
    k_neighbours = KNeighborsClassifier(n_neighbors=k)
    print(rgbs[0])
    k_neighbours.fit(rgbs, labels)

    return k_neighbours

def extract_tags_from(image_list,function="2d",show=False, show_rgb=False, idx_to_see=None):
    '''
    Takes a list of .np image files and extracts the tags from the files
    Then uses the function to fit and guess the tag's colour, height.etc
    
    # Returns
    Guesses - List of successfully fitted guesses
    Indexes - List of tag indexes for the image (order in the json file, requires all images to be labelled in the same order)
    '''
    json_list = to_json_list(image_list)
    count = 0
    fig, axs = plt.subplots(3,4)
    guesses = []
    indexes = []
    not_found = []
    for file in json_list:
        image_indexes = []
        image_guesses = []
        img_raw, tag_list = btcolour.fit.image_loader(file)
        tags = btcolour.fit.get_tags(img_raw,tag_list,buffer=7)

        for i, tag in enumerate(tags):
            found = False
            guess = None
            buffer = 4
            while not found and buffer < 20:
                try:
                    guess = btcolour.fit.fit_from_img(tag,function=function)
                    image_guesses.append(guess)
                    image_indexes.append(i)
                    found = True
                except RuntimeError:
                    found = False
                    buffer += 1
                    tag = btcolour.fit.get_tags(img_raw,tag_list,buffer,indexes=[i])
            if not found:
                #print("tag %s wasn't fitted in image %s", {i, os.path.split(file)[1]})
                #image_guesses.append(None)
                #image_indexes.append(i)
                if i not in not_found: not_found.append(i)

            elif show:
                if count > 11:
                    fig, axs = plt.subplots(3,4)
                    count = 0

                try:
                    axs[count//4,count%4].imshow(tag)
                    axs[count//4,count%4].set_title(i)
                except TypeError:
                    print("BLACK AND WHITE: "+file)
                count += 1
                axs[count//4,count%4].imshow(btcolour.fit.reconstruct(guess,50,buffer,function))
                axs[count//4,count%4].scatter(40,40,c=[colour_from_guess(guess)],s=400)
                count+=1

        indexes.append(image_indexes)
        guesses.append(image_guesses)

    if show_rgb == True:
        plot_guesses(guesses, indexes, i = idx_to_see)

    return guesses, indexes

def ext_indexes(guesses, tag_indxes, inds):
    '''
    Extract specific tags from a list of guesses and indexes
    Assumes all images are labelled in the same way 
    For this data set, the top left is 0, top right is 6 and bottom right is 39

    #Returns
    cut_guesses -
    cut_indexes -
    all_rgbs - extracted, normalised rgbs
    all_indexes - tag indexes matching the rgbs
    '''
    cut_guesses = []
    cut_indexes = []
    for g, i in zip(guesses,tag_indxes):
        if inds != None:
            idx_to_slice = np.in1d(np.array(i),np.array(inds))
            cut_guesses.append(np.array(g)[idx_to_slice])
            cut_indexes.append(np.array(i)[idx_to_slice])

    all_rgbs = []
    all_indexes = []

    for guess, index in zip(cut_guesses, cut_indexes):
        rgbs = [[i[3],i[4],i[5]] for i in guess]
        rgbs = norm_rgbs(rgbs)
        for rgb, i in zip(rgbs, index):
            all_rgbs.append(rgb)
            all_indexes.append(i)

    return cut_guesses, cut_indexes, np.array(all_rgbs), np.array(all_indexes)

def plot_guesses(guesses, indexes,ax=None, i=None):
    if ax == None:
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")


    for guess, index in zip(guesses, indexes):
        rgbs = [[i[3],i[4],i[5]] for i in guess]
        rgbs = norm_rgbs(rgbs)
   
        for label, rgb in zip(index,rgbs):
            if i is not None and label in i:
                (x,y,z) = rgb
                ax.scatter(x,y,z,color=(x,y,z),s=100)
                ax.text(x+0.01,y+0.01,z+0.01,label)
            elif i is None:
                (x,y,z) = rgb
                ax.scatter(x,y,z,color=(x,y,z),s=100)
                ax.text(x+0.01,y+0.01,z+0.01,label)

def best_tag_groups(guesses, indexes):
    stds = []
    means = []

    for i in range(40):
        _, _, rgbs, _ = ext_indexes(guesses,indexes,[i])
        stds.append(np.std(rgbs))
        means.append(np.mean(rgbs,axis=0))

    return np.argsort(np.array(stds)), stds, means