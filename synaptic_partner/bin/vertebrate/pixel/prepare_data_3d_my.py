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


    
    def __init__(self, imagedir, gtname=None):
        
        if gtname!=None:
            fid = h5py.File(gtname)
            gtvol = np.array(fid['stack'])
            fid.close()
            
            gtvol=gtvol.astype(np.float32)
            self.membraneImages= gtvol
        
        allfiles = sorted(glob.glob(imagedir+'/*.png'))
        #pathPrefix = datadir
        #img_search_string_membraneImages = pathPrefix + 'labels/membranes_fullContour/' + purpose + '/'+block_name+'/*.tif'
        #img_search_string_labelImages    = pathPrefix + 'labels/' + purpose + '/'+block_name+'/*.tif'
        #img_search_string_grayImages     = pathPrefix + 'images/' + purpose + '/'+block_name+'/*.tif'

        
        #img_files_gray     = sorted( glob.glob( img_search_string_grayImages ) )
        #img_files_membrane = sorted( glob.glob( img_search_string_membraneImages ) )
        #img_files_labels   = sorted( glob.glob( img_search_string_labelImages ) )


        img = np.array(Image.open(allfiles[0])) #read the first image to get imformation about the shape
        
        self.grayImages    = np.zeros((len(allfiles), img.shape[0], img.shape[1])).astype(np.float32)
        


        #read_order = range(np.shape(img_files_gray)[0])
        #for img_index in read_order:
        for ii,filename in enumerate(allfiles):
            print filename
            img = np.array(Image.open(filename)).astype(np.float32)
            
            img_normalize_toufiq = img *1.0/255

            self.grayImages[ii,:,:] = img_normalize_toufiq
            
            


    def get_3d_sample(self, nsamples_patch=5, nsamples_block=10, patchSize=572, patchSize_out=388, patchZ=7, patchZ_out=1, doAugmentation=False):



    
        start_time = time.time()
        cropSize = (patchSize - patchSize_out)/2
        csZ = (patchZ - patchZ_out)/2

        nsamples=nsamples_block    

        grayImg_set = np.zeros((nsamples, patchZ, patchSize, patchSize))
        membrane_set= np.zeros((nsamples, patchZ_out, patchSize_out, patchSize_out))
        label_set   = np.zeros((nsamples, patchZ_out, patchSize_out, patchSize_out))

        num_total = 0
        
        x_index = random.randint(patchSize/2,self.grayImages.shape[2]-(patchSize/2))
        y_index = random.randint(patchSize/2,self.grayImages.shape[1]-(patchSize/2))
        z_index = random.randint(patchZ/2,self.grayImages.shape[0]-(patchZ/2))
        
        
        grayImg_set[num_total,:,:,:] = self.grayImages[z_index-(patchZ/2):z_index+(patchZ/2), y_index-(patchSize/2):y_index+(patchSize/2), x_index-(patchSize/2):x_index+(patchSize/2)]

        lbl_startz = z_index-(patchZ/2)+csZ
        lbl_endz = lbl_startz +patchZ_out
        lbl_starty = y_index-(patchSize/2)+cropSize
        lbl_endy = lbl_starty + patchSize_out
        lbl_startx = x_index-(patchSize/2)+cropSize
        lbl_endx = lbl_startx+patchSize_out
        
        membrane_set[num_total,:,:,:]= self.membraneImages[lbl_startz:lbl_endz, lbl_starty :lbl_endy, lbl_startx:lbl_endx]
        
        originalNum = num_total
        num_total += 1
        
        grayImg_set = grayImg_set[:num_total,...]
        membrane_set = membrane_set[:num_total,...]
        
        
        reflectz=random.randint(0,1)
        reflecty=random.randint(0,1)
        reflectx=random.randint(0,1)
        swapxy=random.randint(0,1)

        if reflectz:
            grayImg_set = grayImg_set[:,::-1,:,:]
            membrane_set = membrane_set[:,::-1,:,:]

        if reflecty:
            grayImg_set = grayImg_set[:,:,::-1,:]
            membrane_set = membrane_set[:,:,::-1,:]

        if reflectx:
            grayImg_set = grayImg_set[:,:,:,::-1]
            membrane_set = membrane_set[:,:,:,::-1]

        if swapxy:
            grayImg_set = grayImg_set.transpose((0,1,3,2))
            membrane_set = membrane_set.transpose((0,1,3,2))
            

        
        self.scale_low = 0.8
        self.scale_high = 1.2
        self.shift_low = -0.2
        self.shift_high = 0.2


        grayImg_set_mean = grayImg_set.mean()
        grayImg_set = grayImg_set_mean + (grayImg_set -grayImg_set_mean)*np.random.uniform(low=self.scale_low,high=self.scale_high)
        grayImg_set = grayImg_set + np.random.uniform(low=self.shift_low,high=self.shift_high)
        grayImg_set = np.clip(grayImg_set, 0.05, 0.95)

        print "number of labels: "+str(len(np.unique(membrane_set)))

        ##pdb.set_trace()
        #newMembrane = np.zeros((num_total, patchZ_out*patchSize_out*patchSize_out))
        #for i in range(num_total):
            #newMembrane[i] = membrane_set[i].flatten()

        data_set = (grayImg_set, membrane_set)
        

        end_time = time.time()
        total_time = (end_time - start_time)


        return data_set


    def get_test_sample(self, nsamples_patch=5, nsamples_block=10, patchSize=572, patchSize_out=388, patchZ=7, patchZ_out=1, doAugmentation=False):

        
 	pdb.set_trace()       
        
        rangex1 = set(range(patchSize/2, self.grayImages.shape[2]-patchSize/2, patchSize_out))
        rangex1.add(self.grayImages.shape[2]-patchSize/2)
        rangex = sorted(list(rangex1))

        rangey1 = set(range(patchSize/2, self.grayImages.shape[1]-patchSize/2, patchSize_out))
        rangey1.add(self.grayImages.shape[1]-patchSize/2)
        rangey = sorted(list(rangey1))
        
        rangez1 = set(range(patchZ/2, self.grayImages.shape[0]-patchZ/2, patchZ_out))
        rangez1.add(self.grayImages.shape[0]-patchZ/2)
        rangez = sorted(list(rangez1))

	nsamples = len(rangez)*len(rangey)*len(rangex) 
	grayImg_set = np.zeros((nsamples, patchZ, patchSize, patchSize))
       
        count=0
        for z_index in rangez:
            for y_index in rangey:
                for x_index in  rangex:
                    grayImg_set[count,:,:,:] = self.grayImages[z_index-(patchZ/2):z_index+(patchZ/2), y_index-(patchSize/2):y_index+(patchSize/2), x_index-(patchSize/2):x_index+(patchSize/2)]
                    
                    count =count+1
                    
        grayImg_set = grayImg_set[:count,...]
        return grayImg_set
                    
    def compute_3d_result(self, prediction1D, savename, nsamples_patch=5, nsamples_block=10, patchSize=572, patchSize_out=388, patchZ=7, patchZ_out=1, doAugmentation=False):

        cropSize = (patchSize - patchSize_out)/2
        csZ = (patchZ - patchZ_out)/2


        result_vol = np.zeros(self.grayImages.shape)

        count=0
        
        rangex1 = set(range(patchSize/2, self.grayImages.shape[2]-patchSize/2, patchSize_out))
        rangex1.add(self.grayImages.shape[2]-patchSize/2)
        rangex = sorted(list(rangex1))

        rangey1 = set(range(patchSize/2, self.grayImages.shape[1]-patchSize/2, patchSize_out))
        rangey1.add(self.grayImages.shape[1]-patchSize/2)
        rangey = sorted(list(rangey1))
        
        rangez1 = set(range(patchZ/2, self.grayImages.shape[0]-patchZ/2, patchZ_out))
        rangez1.add(self.grayImages.shape[0]-patchZ/2)
        rangez = sorted(list(rangez1))
        
        for z_index in rangez:
            for y_index in rangey:
                for x_index in  rangex:
                    
                    
                    result_reshaped = prediction1D[count,...] 
                    result_reshaped = result_reshaped.reshape((patchZ_out, patchSize_out, patchSize_out))
                    
                    lbl_startz = z_index-(patchZ/2)+csZ
                    lbl_endz = lbl_startz +patchZ_out
                    lbl_starty = y_index-(patchSize/2)+cropSize
                    lbl_endy = lbl_starty + patchSize_out
                    lbl_startx = x_index-(patchSize/2)+cropSize
                    lbl_endx = lbl_startx+patchSize_out
                    
                    result_vol[lbl_startz:lbl_endz, lbl_starty:lbl_endy, lbl_startx:lbl_endx] =  result_reshaped
                    
                    count =count+1
                    
        fidw=h5py.File(savename,'w')
        fidw.create_dataset('stack',data=result_vol.astype(np.float32))
        fidw.close()

