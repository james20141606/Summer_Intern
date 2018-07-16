#! /usr/bin/env python
import h5py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

with h5py.File('data/sample_A_padded_20160501.hdf') as f:
    volumes_labels_clefts  =f['volumes/labels/clefts'][:]
    volumes_labels_neuron_ids  =  f['volumes/labels/neuron_ids'][:]
    volumes_raw = f['volumes/raw'][:]




imagelist = [volumes_raw[i] for i in range(125)]
fig = plt.figure(figsize=(6,6)) # make figure

# make axesimage object
# the vmin and vmax here are very important to get the color map correct

im =plt.imshow(imagelist[0], cmap='gist_earth')

def updatefig(j):
    # set the data in the axesimage object
    im.set_array(imagelist[j])
    # return the artists set
    return [im]
# kick off the animation
anim = animation.FuncAnimation(fig, updatefig, frames=range(125),
                               interval=40, blit=True)


anim.save('animation.mp4', writer='imagemagick', fps=25)
