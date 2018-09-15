import time
import glob
#import mahotas
import numpy as np
import scipy
import scipy.misc
import random
from keras.models import Model, Sequential, model_from_json
from PIL import Image
import h5py

import pdb


class GenerateData:

    def __init__(self, imagedir, patchSize, patchSize_out, patchZ, patchZ_out, gtname=None):

        if gtname != None:

            fid = h5py.File(gtname)
            gtvol = np.array(fid['stack'])
            fid.close()

            gtvol = gtvol.astype(np.float32)
            self.membraneImages = gtvol

        allfiles = sorted(glob.glob(imagedir+'/*.png'))

        self.patchSize = patchSize
        self.patchSize_out = patchSize_out
        self.patchZ = patchZ
        self.patchZ_out = patchZ_out

        #pathPrefix = datadir
        #img_search_string_membraneImages = pathPrefix + 'labels/membranes_fullContour/' + purpose + '/'+block_name+'/*.tif'
        #img_search_string_labelImages    = pathPrefix + 'labels/' + purpose + '/'+block_name+'/*.tif'
        #img_search_string_grayImages     = pathPrefix + 'images/' + purpose + '/'+block_name+'/*.tif'

        #img_files_gray     = sorted( glob.glob( img_search_string_grayImages ) )
        #img_files_membrane = sorted( glob.glob( img_search_string_membraneImages ) )
        #img_files_labels   = sorted( glob.glob( img_search_string_labelImages ) )

        # read the first image to get imformation about the shape
        img = np.array(Image.open(allfiles[0]))

        self.grayImages = np.zeros(
            (len(allfiles), img.shape[0], img.shape[1])).astype(np.float32)

        #read_order = range(np.shape(img_files_gray)[0])
        #for img_index in read_order:
        for ii, filename in enumerate(allfiles):
            print filename
            img = np.array(Image.open(filename)).astype(np.float32)

            img_normalize_toufiq = img * 1.0/255

            self.grayImages[ii, :, :] = img_normalize_toufiq

    def compute_test_sample_indices(self):

        rangex1 = set(range(
            self.patchSize/2, self.grayImages.shape[2]-self.patchSize/2, self.patchSize_out))
        rangex1.add(self.grayImages.shape[2]-self.patchSize/2-1)
        rangex = sorted(list(rangex1))

        rangey1 = set(range(
            self.patchSize/2, self.grayImages.shape[1]-self.patchSize/2, self.patchSize_out))
        rangey1.add(self.grayImages.shape[1]-self.patchSize/2-1)
        rangey = sorted(list(rangey1))

        rangez1 = set(
            range(self.patchZ/2, self.grayImages.shape[0]-self.patchZ/2, self.patchZ_out))
        rangez1.add(self.grayImages.shape[0]-self.patchZ/2-1)
        rangez = sorted(list(rangez1))

	nsamples = len(rangez)*len(rangey)*len(rangex)
	self.test_sample_indices = np.zeros((nsamples, 3)).astype(np.uint32)

        count = 0
        for z_index in rangez:
            for y_index in rangey:
                for x_index in rangex:

                    self.test_sample_indices[count, :] = [
                        z_index, y_index, x_index]

                    count = count+1

        self.test_sample_indices = self.test_sample_indices[:count, ...]
        return count

    def get_next_test_sample(self, idx):

        z_index, y_index, x_index = self.test_sample_indices[idx, :]

        return self.grayImages[z_index-(self.patchZ/2):z_index+(self.patchZ/2), y_index-(self.patchSize/2):y_index+(self.patchSize/2), x_index-(self.patchSize/2):x_index+(self.patchSize/2)]

    def create_result_vol(self):

        self.result_vol = np.zeros(self.grayImages.shape)

    def write_prediction_vol(self, idx, pred):
        z_index, y_index, x_index = self.test_sample_indices[idx, :]
        cropSize = (self.patchSize - self.patchSize_out)/2
        csZ = (self.patchZ - self.patchZ_out)/2

        result_reshaped = pred
        result_reshaped = result_reshaped.reshape(
            (self.patchZ_out, self.patchSize_out, self.patchSize_out))

        lbl_startz = z_index-(self.patchZ/2)+csZ
        lbl_endz = lbl_startz + self.patchZ_out
        lbl_starty = y_index-(self.patchSize/2)+cropSize
        lbl_endy = lbl_starty + self.patchSize_out
        lbl_startx = x_index-(self.patchSize/2)+cropSize
        lbl_endx = lbl_startx+self.patchSize_out

        self.result_vol[lbl_startz:lbl_endz, lbl_starty:lbl_endy,
                        lbl_startx:lbl_endx] = result_reshaped
        print "label ({0},{1},{2}) ".format(lbl_endz, lbl_endy, lbl_endx)

    def save_to_disk(self, savename):
        #pdb.set_trace()
        marginxy = (self.patchSize-self.patchSize_out)/2
        marginz = (self.patchZ-self.patchZ_out)/2
        if not (marginxy == 0) & (marginz == 0):
            resultvol = self.result_vol[marginz:-marginz,
                                        marginxy:-marginxy, marginxy:-marginxy]
        elif (marginxy == 0) & (marginz == 0):
            resultvol = self.result_vol
            print ('identity')
        print (marginxy, marginz)
        #resultvol = self.result_vol[marginz:-marginz,
        #                            marginxy:-marginxy, marginxy:-marginxy]
        fidw = h5py.File(savename, 'w')
        fidw.create_dataset('stack', data=resultvol.astype(np.float32))
        fidw.close()
        print ('savedatshape: ',aresultvol.shape)
