import time
import glob
#import mahotas
import numpy as np
import scipy
import scipy.misc
import random
from PIL import Image
import h5py
from scipy.ndimage import gaussian_filter, label, find_objects, distance_transform_edt
import pickle
from rotate3d import *
import itertools
import pdb
from scipy.spatial import KDTree
import os, sys
import argparse


def relabel_from_one(a):
    labels = np.unique(a)
    labels0 = labels[labels!=0]
    m = labels.max()
    if m == len(labels0): # nothing to do, already 1...n labels
        return a, labels, labels
    forward_map = np.zeros(m+1, int)
    forward_map[labels0] = np.arange(1, len(labels0)+1)
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = labels
    return forward_map[a], forward_map, inverse_map



class GenerateData:
    
    def __init__(self, trial_name, datadir, imagedir, predname, segname, syn_gtname=None, seg_gtname=None, patchSize=250, patchZ=24, margin=False, ignore=False):
        #pdb.set_trace()    
        
        fid = h5py.File(os.path.join(datadir,predname))
        self.predvol = np.array(fid['stack'])
        fid.close()
        #self.predvol = self.predvol[14:-14,44:-44,44:-44]
        
        #if margin:
            #self.predvol[:14,...]=0.0
            #self.predvol[-15:,...]=0.0
        
        
        self.segname = segname
        fid = h5py.File(os.path.join(datadir,segname))
        self.segvol = np.array(fid['stack'])
        fid.close()
        
        if syn_gtname!=None:
            fid = h5py.File(os.path.join(datadir,syn_gtname))
            syn_gtvol = np.array(fid['stack'])
            fid.close()
            
            self.syn_gtvol=syn_gtvol.astype(np.int32)
            
        #if ignore:
            #self.syn_gtvol[self.syn_gtvol>200]=0 #not sure ones
        #pdb.set_trace()
        if seg_gtname!=None:
            fid = h5py.File(os.path.join(datadir,seg_gtname))
            seg_gtvol = np.array(fid['stack'])
            fid.close()
            
            self.seg_gtvol=seg_gtvol.astype(np.int32)
        else: 
            self.seg_gtvol = self.segvol

        allfiles = sorted(glob.glob(os.path.join(datadir,imagedir+'/*.png')))
        #pathPrefix = datadir
        #img_search_string_membraneImages = pathPrefix + 'labels/membranes_fullContour/' + purpose + '/'+block_name+'/*.tif'
        #img_search_shttps://arxiv.org/abs/1708.02599tring_labelImages    = pathPrefix + 'labels/' + purpose + '/'+block_name+'/*.tif'
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
        
        #pdb.set_trace()
        
        self.patchSize = patchSize
        self.patchZ = patchZ
        self.trial_name = trial_name
        self.datadir = datadir
        
        if not os.path.exists(self.trial_name):
            os.makedirs(self.trial_name)
        
        
        #self.compute_cc_partners()
        #self.compute_gt_partners()
        
        #pdb.set_trace()
        self.cc_partners = pickle.load(open(os.path.join(self.trial_name,'saved_cc_partners.pkl'),'rb'))
        #self.cc3d = pickle.load(open(os.path.join(self.trial_name,'saved_cc_vol.pkl'),'rb'))
        self.gt_partners = pickle.load(open(os.path.join(self.trial_name,'saved_gt_partners.pkl'),'rb'))
        self.gt_cc3d = pickle.load(open(os.path.join(self.trial_name,'saved_gtcc_vol.pkl'),'rb'))       
        self.find_positive_cc(margin=margin, ignore=ignore)
        
        #self.cc_partners = pickle.load(open(os.path.join(self.trial_name,'saved_cc_partners_with_label.pkl'),'rb'))
        #self.cc_partner_keys = self.cc_partners.keys()
        
        #self.separate_class()

    def separate_class(self):
        
        #pdb.set_trace()
        
        self.positive_examples=[]
        self.negative_examples=[]
        for sid in self.cc_partner_keys:
            set1 = self.cc_partners[sid]
            for pid, entry in enumerate(set1):
                if entry['label']==1:
                    self.positive_examples.append(entry)
                else:
                    self.negative_examples.append(entry)
            
            
            
            
    def compute_cc(self, binary_input, min_size_3d=100):
        
        strel = np.array([[[False, False, False],
                       [False, True, False],
                       [False, False, False]],
                      [[False, True, False],
                       [True,  True, True],
                       [False, True, False]],
                      [[False, False, False],
                       [False, True, False],
                       [False, False, False]]])
                      
        cc3d, count = label(binary_input, strel)
        
        areas = np.bincount(cc3d.ravel())
        ignore = areas < min_size_3d
        ignore_locations = ignore[cc3d]
        cc3d[ignore_locations]=0
        unique_ccid = np.setdiff1d(np.unique(cc3d),[0])
        
        seg_cc_match = scipy.sparse.csc_matrix((np.ones_like(cc3d.ravel()), (cc3d.ravel(), self.segvol.ravel())))
        
        cc_id = np.amax(unique_ccid)+1
        
        seg_overlap = {}
        cc_overlap = {}
        #pdb.set_trace()
        for i in range(unique_ccid.shape[0]):
            uid = unique_ccid[i]
            print 'cc: ',uid
            match_vector = seg_cc_match[uid,:]
            match_segid = np.setdiff1d(np.array((match_vector>min_size_3d).nonzero()[1]),[0])
            
            cc_mask = (cc3d==uid)
            for si,segid in enumerate(match_segid):
                post_seg_mask = (cc_mask * (self.segvol==segid) )>0
                locations = np.array((post_seg_mask>0).nonzero()).T.astype(np.float32)
                
                cc3d[post_seg_mask] = cc_id
                seg1 = {'seg': segid, 'loc': locations}
                seg_overlap[cc_id] = seg1 # cc_id changes in every iteration
                
                
                if segid in cc_overlap.keys():
                    cc_overlap[segid].append(cc_id)
                else:
                    #pdb.set_trace()
                    cc_overlap[segid] = [cc_id]
                
                cc_id = cc_id +1
                
        return cc3d, seg_overlap, cc_overlap

    def read_rag_neighbors(self):
        #pdb.set_trace()
        #if not os.path.exists(os.path.join(self.datadir,'rag_edges.txt')):
        command_str = './synaptic_partner -watershed  '+os.path.join(self.datadir, self.segname)+ '  stack' 
        os.system(command_str)
        os.system('mv  rag_edges.txt '+self.datadir)
            
        fp = open(os.path.join(self.datadir,'rag_edges.txt'))
        lines = fp.readlines()
        rag_nbrs={}
        for line in lines:
            nbr1 = []
            nbr1.append([int(v) for v in line.split()])
            nbr1 = np.squeeze(nbr1)
            node1 = nbr1[0]
            nbr1 = nbr1[1:]
            rag_nbrs[node1] = nbr1
        return rag_nbrs

    def compute_cc_partners(self,threshold=0.3):
        
        mode = 'save'
        rag_neighbors = self.read_rag_neighbors()
        #pdb.set_trace()
        ppfull=self.predvol
        if mode=='save':
            post=abs(ppfull)*(ppfull<-threshold)
            post_mask = post>0
            
            cc3d_post, post_segoverlap, post_ccoverlap= self.compute_cc(post_mask)
            #pdb.set_trace()
            pickle.dump(post_segoverlap, open(os.path.join(self.trial_name,'saved_cc_post_seg.pkl'),'wb'))
            pickle.dump(post_ccoverlap, open(os.path.join(self.trial_name,'saved_seg_post_cc.pkl'),'wb'))
            pickle.dump(cc3d_post, open(os.path.join(self.trial_name,'saved_post_cc3d.pkl'),'wb'))
            
            pre=ppfull*(ppfull>threshold)
            pre_mask = pre>0
            
            cc3d_pre, pre_segoverlap, pre_ccoverlap = self.compute_cc(pre_mask)
            #pdb.set_trace()
            pickle.dump(pre_segoverlap, open(os.path.join(self.trial_name,'saved_cc_pre_seg.pkl'),'wb'))
            pickle.dump(pre_ccoverlap, open(os.path.join(self.trial_name,'saved_seg_pre_cc.pkl'),'wb'))
            pickle.dump(cc3d_pre, open(os.path.join(self.trial_name,'saved_pre_cc3d.pkl'),'wb'))
            
        else:
            
            post_segoverlap = pickle.load(open(os.path.join(self.trial_name,'saved_cc_post_seg.pkl'),'rb'))
            post_ccoverlap = pickle.load(open(os.path.join(self.trial_name,'saved_seg_post_cc.pkl'),'rb'))
            cc3d_post = pickle.load(open(os.path.join(self.trial_name,'saved_post_cc3d.pkl'),'rb'))
            
            pre_segoverlap = pickle.load(open(os.path.join(self.trial_name,'saved_cc_pre_seg.pkl'),'rb'))
            pre_ccoverlap = pickle.load(open(os.path.join(self.trial_name,'saved_seg_pre_cc.pkl'),'rb'))
            cc3d_pre = pickle.load(open(os.path.join(self.trial_name,'saved_pre_cc3d.pkl'),'rb'))
        
        all_post_ccid = post_segoverlap.keys()
        all_candidates={}
        count = 0
        
        seg_szs = np.bincount(self.segvol.ravel())
        SEG_SZ_THD=1000
        #pdb.set_trace()
        for post_cc in all_post_ccid:
            
            print 'post cc:', post_cc
            post_segid = post_segoverlap[post_cc]['seg']
            
            if seg_szs[post_segid]<SEG_SZ_THD:
                continue
            
            if post_segid in rag_neighbors.keys():
                post_segid_nbrs = rag_neighbors[post_segid]
            else:
                print 'node not in rag neighbors'
                continue
        
            for post_seg_nbr in post_segid_nbrs:
                
                if seg_szs[post_seg_nbr]<SEG_SZ_THD:
                    continue
                if post_seg_nbr in pre_ccoverlap.keys():
                    
                    pre_candidates_for_post = pre_ccoverlap[post_seg_nbr]
                    
                    for pre_candidate_for_post in pre_candidates_for_post:
                        

                        
                        pre_loc = pre_segoverlap[pre_candidate_for_post]['loc']
                        post_loc = post_segoverlap[post_cc]['loc']
                        
                        pre_loc_mean = np.mean(pre_loc ,axis=0)
                        post_loc_mean = np.mean(post_loc ,axis=0)
                        
                        pre_loc_mean[0] = pre_loc_mean[0]*(30./4)
                        post_loc_mean[0] = post_loc_mean[0]*(30./4)
                        
                        dist_initial = np.sqrt(np.sum((pre_loc_mean - post_loc_mean)**2))
                        if dist_initial > 200: # initial check to make the computation faster
                            continue;
                        
                        #pdb.set_trace()
                        pre_min = np.min(pre_loc, axis=0)
                        post_min = np.min(post_loc, axis=0)
                        
                        contained_min = np.min(np.stack((pre_min,post_min),axis=0), axis=0).astype(np.uint32)
                        
                        pre_max = np.max(pre_loc, axis=0)
                        post_max = np.max(post_loc, axis=0)
                        
                        contained_max = np.max(np.stack((pre_max,post_max),axis=0), axis=0).astype(np.uint32) + 1
                        
                        contained_cube_post = cc3d_post[contained_min[0]:contained_max[0], contained_min[1]:contained_max[1], contained_min[2]:contained_max[2]]
                        contained_cube_post = (contained_cube_post== post_cc)
                        
                        contained_cube_pre = cc3d_pre[contained_min[0]:contained_max[0], contained_min[1]:contained_max[1], contained_min[2]:contained_max[2]]
                        
                        contained_cube_dt = distance_transform_edt(np.invert(contained_cube_post), sampling=[30.0,4.0,4.0])
                        dist = contained_cube_dt[contained_cube_pre==pre_candidate_for_post]
                        #dist = np.amin(cdist)
                        SEP_DIST_THD=50
                        if np.amin(dist)<=SEP_DIST_THD:
                            
                            #pdb.set_trace()
                            max_dist=SEP_DIST_THD #np.amax(np.sort(dist,axis=None)[:40])
                            constr1=(contained_cube_dt<=max_dist)*(contained_cube_pre==pre_candidate_for_post)
                            contact_loc1 = np.array((constr1>0).nonzero()).T
                            
                            dist1_consider=contained_cube_dt[contact_loc1[:,0],contact_loc1[:,1],contact_loc1[:,2]]
                            sorted_dist_idx = np.argsort(dist1_consider,axis=None)
                            
                            contact_loc1_sorted_dist = contact_loc1[sorted_dist_idx,:]
                            
                            contact_loc = contact_loc1_sorted_dist + np.repeat(np.array([contained_min]),contact_loc1_sorted_dist.shape[0],axis=0)
                            
                            
                            #contact_loc = contact_loc1 + np.repeat(np.array([contained_min]),contact_loc1.shape[0],axis=0)
                            
                            candidate1 = {'pre':pre_candidate_for_post, 'post':post_cc, 'post_seg': post_segid, 'pre_seg': post_seg_nbr, 'contact_loc': contact_loc}
                            if post_cc in all_candidates.keys():
                                all_candidates[post_cc].append(candidate1)
                            else:
                                all_candidates[post_cc] = [candidate1]
                                
                            count = count +1
                            print "dist: {0}, pre: {1}, post: {2}".format(np.amin(dist), pre_candidate_for_post, post_cc)
                            #print "pre location: ", pre_loc
                            #print "post location: ", post_loc
                            #pdb.set_trace()
                            
        
        
        pickle.dump(all_candidates, open(os.path.join(self.trial_name,'saved_cc_partners.pkl'),'wb'))
        #pickle.dump(self.cc3d, open('saved_cc_vol.pkl','wb'))            
        
        
        return np.array(all_candidates)

    def compute_gt_partners(self):
        
        #pdb.set_trace()
        
        pre = (self.syn_gtvol%2)==1
        post = (self.syn_gtvol>0)*((self.syn_gtvol%2)==0)
        gtcc3d = self.syn_gtvol
        gtcc3d[post==1] = gtcc3d[post==1]-1
        unique_gtid = np.setdiff1d(np.unique(gtcc3d),[0])
        
        
        seg_gt_match = scipy.sparse.csc_matrix((np.ones_like(gtcc3d.ravel()), (gtcc3d.ravel(), self.seg_gtvol.ravel())))
        
        
        cc_ctrs = {}
        for i in range(unique_gtid.shape[0]):
            #if ignore[i]==False:
            uid = unique_gtid[i]
            print 'synapse gt: ',uid
            match_vector = seg_gt_match[uid,:]
            match_segid = np.setdiff1d(np.array((match_vector>20).nonzero()[1]),[0])
            
            #if len(match_segid)<2:
                #continue
            cc_mask = (gtcc3d == uid)
            overlap_count = np.zeros((2,len(match_segid)))
            for si,segid in enumerate(match_segid):
                seg_mask = (self.seg_gtvol==segid)
                overlap_count[0,si] = np.sum(pre*seg_mask*cc_mask)
                overlap_count[1,si] = np.sum(post*seg_mask*cc_mask)
            
            count_sum=np.sum(overlap_count,axis=1)
            divisor = np.tile(np.matrix(count_sum),(overlap_count.shape[1],1)).T
            overlap_pct = overlap_count/(divisor+1)
            
            
            post_candidate =  match_segid[np.argmax(overlap_pct[1,...])]
            if len(match_segid)>=2:
                overlap_pct[:,np.argmax(overlap_pct[1,...])]=0
                pre_candidate =  match_segid[np.argmax(overlap_pct[0,...])]
            else:
                #pdb.set_trace()
                print 'gt invalid'
                gtcc3d[gtcc3d==uid]=0
                continue;
            #if len(pre_candidates)<1 or len(post_candidates)<1:
                #pdb.set_trace()
                #print 'gt invalid_pair'
                #continue;
            
            #match_candidates = np.array([[x,y] for x in pre_candidates for y in post_candidates])
            
            #valid_idx=(match_candidates[:,0]!=match_candidates[:,1]).nonzero()[0]
            #match_candidates2 = match_candidates[valid_idx,:]
            
            #if pre_candidate == post_candidate:
            
            ctr1 = {'pre': pre_candidate, 'post': post_candidate}
            
            cc_ctrs[uid] = ctr1
            
        #pdb.set_trace()
        pickle.dump(cc_ctrs, open(os.path.join(self.trial_name,'saved_gt_partners.pkl'),'wb'))
        pickle.dump(gtcc3d, open(os.path.join(self.trial_name,'saved_gtcc_vol.pkl'),'wb'))
     
    
    def find_positive_cc(self,margin=False, ignore=False):
        
        #cc3d = self.cc3d[14:-14,44:-44,44:-44]
        gt=self.gt_cc3d
        if margin:
            gt[14:-14,44:-44,44:-44]=0
        
        if ignore:
            gt[gt>200]=0
       
        cc3d = pickle.load(open(os.path.join(self.trial_name,'saved_post_cc3d.pkl'),'rb'))
        #post = (self.gtvol)*((self.gtvol%2)==0)
        #gt = post
        #cc3d = cc3d[11:-11,192/2:-192/2,192/2:-192/2]
        #gt = gt[11:-11,192/2:-192/2,192/2:-192/2]
        
        #pdb.set_trace()

        seg_gt_overlap = scipy.sparse.csc_matrix((np.ones_like(self.segvol.ravel()), (self.segvol.ravel(), self.seg_gtvol.ravel())))
        seg_gt_map = np.argmax(seg_gt_overlap[:,1:],axis=1)
        
        
        overlaps= scipy.sparse.csc_matrix((np.ones_like(gt.ravel()), (gt.ravel(), cc3d.ravel())))
        from scipy.sparse import find
        ntp = 0
        nfp = 0
        count=0
        #unique_ccid = np.unique(cc3d)
        #for ii in range(1,unique_ccid.shape[0]):
            #uid = unique_ccid[ii]
        #pdb.set_trace()
        true_positive_detected=[]
        detection_true=0
        detection_false=0
        for uid in self.cc_partners.keys():
            
	    rows, dummy, values = find(overlaps[:,uid])
            count=count+len(self.cc_partners[uid])
            if rows[0]==0:
                rows = rows[1:]
                values = values[1:]
            if len(values)<1:
                for ci in range(len(self.cc_partners[uid])):
                    self.cc_partners[uid][ci]['label'] = -1
                
                detection_false = detection_false + len(self.cc_partners[uid])
                continue
            
            #match_id = rows[np.argmax(values)]
            found=0
            for mi in range(rows.shape[0]):
                match_id = rows[mi]
                match_amt = values[mi]
                #print 'found: {0}, {1}'.format(self.cc_partners[uid]['pre'], self.cc_partners[uid]['post'])
                #print 'gt: {0}, {1}'.format(self.gt_partners[match_id]['pre'], self.gt_partners[match_id]['post'])
                print uid, match_id
		#if uid==3805 and match_id==185:
		    #pdb.set_trace()
		for ci in range(len(self.cc_partners[uid])):
                    
                    pre_seg_gt_id = seg_gt_map[self.cc_partners[uid][ci]['pre_seg']][0,0]+1
                    post_seg_gt_id = seg_gt_map[self.cc_partners[uid][ci]['post_seg']][0,0]+1
                    if pre_seg_gt_id ==0 or post_seg_gt_id==0:
                        print "seg id not present in gt"
                        #pdb.set_trace()
                        continue
                    
                    if (pre_seg_gt_id == self.gt_partners[match_id]['pre']) and \
                    (post_seg_gt_id == self.gt_partners[match_id]['post']):
                        
                    #print 'true positive ', uid
                        print 'found: {0}, {1}'.format(self.cc_partners[uid][ci]['pre_seg'], self.cc_partners[uid][ci]['post_seg'])
                        print 'gt: {0}, {1}'.format(self.gt_partners[match_id]['pre'], self.gt_partners[match_id]['post'])
                        
                        true_positive_detected.append(match_id)
                        detection_true = detection_true+1
                        
                        self.cc_partners[uid][ci]['label'] = 1
                        self.cc_partners[uid][ci]['gt_id'] = match_id
                        
                        
                    else:
                       if self.cc_partners[uid][ci].has_key('label'):
                           if self.cc_partners[uid][ci]['label']==1:
                                continue;
                       else:
                        #break
  			    detection_false = detection_false + 1
                            self.cc_partners[uid][ci]['label'] = -1
                            self.cc_partners[uid][ci]['gt_id'] = match_id
                        #break
                
        
        unique_gt = np.setdiff1d(np.unique(gt),[0])
        nunique_detected = len(np.unique(true_positive_detected))

        nmiss = len(unique_gt)-nunique_detected

        print 'detected = ',nunique_detected
        print 'missed =  {0} ({1})'.format(nmiss, nmiss*1./len(unique_gt))
        print 'detection_true = ',detection_true
        print 'detection false = ',detection_false
        
        #pdb.set_trace()
        pickle.dump(self.cc_partners, open(os.path.join(self.trial_name,'saved_cc_partners_with_label.pkl'),'wb'))
        #overlapb=overlaps[1:,:]>20
        #mcolsum = np.sum(overlapb,axis=0)
        #true_positive_idx = (mcolsum>0).A1
        
        #return true_positive_idx
        
    def compute_cube(self,partner_entry):
        #pdb.set_trace()
        patchZ = self.patchZ
        patchSize = self.patchSize
        patchZ_half = patchZ/2
        patchSize_half = patchSize/2
        count=0
        #for uid in self.cc_partners.keys():
            #for ci in range(len(self.cc_partners[uid])):
        post_id = partner_entry['post']
        post_seg = partner_entry['post_seg']
        pre_id = partner_entry['pre']
        pre_seg = partner_entry['pre_seg']
        
        #random_locid = random.sample(range(len(partner_entry['contact_loc'])),1)[0]
        contact_loc = partner_entry['contact_loc'][5]
        
        z_disp = random.sample([-2,0,2], 1)[0]
        y_disp = random.sample([-10,0,10], 1)[0]
        x_disp = random.sample([-10,0,10], 1)[0]
        
        contact_loc = contact_loc + np.array([z_disp, y_disp, x_disp])
        
        zmin = max(0,contact_loc[0]-patchZ_half)
        zmax = min(contact_loc[0]+patchZ_half,self.predvol.shape[0])
        ymin = max(0,contact_loc[1]-patchSize_half)
        ymax = min(contact_loc[1]+patchSize_half,self.predvol.shape[1])
        xmin = max(0,contact_loc[2]-patchSize_half)
        xmax = min(contact_loc[2]+patchSize_half,self.predvol.shape[2])
        
        bzmin = max(0,patchZ_half-contact_loc[0])
        bzmax = patchZ - max(0, (contact_loc[0]+patchZ_half)-self.predvol.shape[0])
        bymin = max(0,patchSize_half-contact_loc[1])
        bymax = patchSize - max(0, (contact_loc[1]+patchSize_half)-self.predvol.shape[1])
        bxmin = max(0,patchSize_half-contact_loc[2])
        bxmax = patchSize - max(0, (contact_loc[2]+patchSize_half)-self.predvol.shape[2])
        
        predvol = np.zeros((patchZ,patchSize,patchSize)).astype(np.float32)
        predvol[bzmin:bzmax, bymin:bymax, bxmin:bxmax] =  self.predvol[zmin:zmax, ymin:ymax, xmin:xmax]
        
        segvol_pre = np.zeros((patchZ,patchSize,patchSize)).astype(np.float32)
        segvol_pre[bzmin:bzmax, bymin:bymax, bxmin:bxmax] = self.segvol[zmin:zmax, ymin:ymax, xmin:xmax]
        segvol_pre = (segvol_pre==pre_seg).astype(np.float32)
        
        segvol_post = np.zeros((patchZ,patchSize,patchSize)).astype(np.float32)
        segvol_post[bzmin:bzmax, bymin:bymax, bxmin:bxmax] = self.segvol[zmin:zmax, ymin:ymax, xmin:xmax]
        segvol_post = (segvol_post==post_seg).astype(np.float32)

        imgvol = np.zeros((patchZ,patchSize,patchSize)).astype(np.float32)
        imgvol[bzmin:bzmax, bymin:bymax, bxmin:bxmax] = self.grayImages[zmin:zmax, ymin:ymax, xmin:xmax]
                
                #count=count+1
                #fid = h5py.File('samples/sample_'+str(count).zfill(5)+'_label'+str(self.cc_partners[post_id][ci]['label'])+'.h5','w')
                #fid.create_dataset('imgvol',data=imgvol)
                #fid.create_dataset('predvol',data=predvol)
                #fid.create_dataset('segvol_pre',data=segvol_pre)
                #fid.create_dataset('segvol_post',data=segvol_post)
                #fid.close()
                #pdb.set_trace()
        return imgvol, predvol, segvol_pre,segvol_post
    
    def get_3d_sample(self, nsamples_batch=5 ):

        #pdb.set_trace()
        nsamples=nsamples_batch    
        
        random_idx = random.sample(self.cc_partner_keys, nsamples)
        
        input_set = np.zeros((nsamples, 4, self.patchZ, self.patchSize, self.patchSize))
        label_set = np.zeros(nsamples).astype(np.int32)
        
        for i,idx in enumerate(random_idx):
            npt = len(self.cc_partners[idx])
            random_idx_j = random.sample(range(npt),1)[0]
            
            combined = self.compute_cube(idx, random_idx_j)
            
            combined = self.reflect_swap(combined)
            
            #imgvol1 = self.scale_intensity(imgvol1)
            
            combined = self.rotate3d(combined)
            
            input_set[i,:,:,:,:] = combined
            label_set[i] = self.cc_partners[idx][random_idx_j]['label']
            
            
    
        #start_time = time.time()
        
        

        print "number of positive labels: "+str(np.sum(label_set>0))


        data_set = (input_set, label_set)
        

        #end_time = time.time()
        #total_time = (end_time - start_time)


        return data_set

    def get_3d_sample2(self, nsamples_batch=5 ):

        #pdb.set_trace()
        nsamples=nsamples_batch    
        npositive = 5
        nnegative = nsamples-npositive
        
        
        input_set = np.zeros((nsamples, 4, self.patchZ, self.patchSize, self.patchSize))
        label_set = np.zeros(nsamples).astype(np.int32)
        
        random_pidx = random.sample(range(len(self.positive_examples)), npositive)
        random_nidx = random.sample(range(len(self.positive_examples)), nnegative)

        count=0
        for pidx in random_pidx:
            pentry = self.positive_examples[pidx]
            combined = self.compute_cube(pentry)
            
            combined = self.reflect_swap(combined)
            
            #imgvol1 = self.scale_intensity(imgvol1)
            
            combined = self.rotate3d(combined)
            
            input_set[count,:,:,:,:] = combined
            label_set[count] = pentry['label']
            count=count+1
            
        for nidx in random_nidx:
            pentry = self.negative_examples[nidx]
            combined = self.compute_cube(pentry)
            
            combined = self.reflect_swap(combined)
            
            #imgvol1 = self.scale_intensity(imgvol1)
            
            combined = self.rotate3d(combined)
            
            input_set[count,:,:,:,:] = combined
            label_set[count] = pentry['label']
            count=count+1
            
            
    
        #start_time = time.time()
        
        

        print "number of positive labels: "+str(np.sum(label_set>0))

        data_set = (input_set, label_set)
        

        #end_time = time.time()
        #total_time = (end_time - start_time)


        return data_set

    def rotate3d(self, input_vol):

        #pdb.set_trace()
        nrotate = random.randint(0,3)
        axis = 0 #random.randint(0,2)
        
        output_vol=[]
        for i in range(len(input_vol)):
            grayImg_set = input_vol[i]
            grayImg_set = axial_rotations(grayImg_set, rot=nrotate, ax=axis)
        
            output_vol.append([])
            output_vol[-1] = grayImg_set
            
        return output_vol
    
    def reflect_swap(self, input_vol):

        
        reflectz=random.randint(0,1)
        reflecty=random.randint(0,1)
        reflectx=random.randint(0,1)
        swapxy=random.randint(0,1)

        output_vol=[]
        for i in range(len(input_vol)):
            grayImg_set = input_vol[i]
            if reflectz:
                grayImg_set = grayImg_set[::-1,:,:]

            if reflecty:
                grayImg_set = grayImg_set[:,::-1,:]

            if reflectx:
                grayImg_set = grayImg_set[:,:,::-1]

            if swapxy:
                grayImg_set = grayImg_set.transpose((0,2,1))
            
            output_vol.append([])
            output_vol[-1] = grayImg_set
            
        return output_vol
    
    
        
    def scale_intensity(self, input_vol):
            
        grayImg_set = input_vol
            
        self.scale_low = 0.8
        self.scale_high = 1.2
        self.shift_low = -0.2
        self.shift_high = 0.2


        grayImg_set_mean = grayImg_set.mean()
        grayImg_set = grayImg_set_mean + (grayImg_set -grayImg_set_mean)*np.random.uniform(low=self.scale_low,high=self.scale_high)
        grayImg_set = grayImg_set + np.random.uniform(low=self.shift_low,high=self.shift_high)
        grayImg_set = np.clip(grayImg_set, 0.05, 0.95)
        
        return grayImg_set




parser = argparse.ArgumentParser(description='Generate syn partner candidates...')
parser.add_argument('--trial', dest='train_trial', action='store', default='trial00', help='trial id')
parser.add_argument('--datadir', dest='datadir', action='store', default='.', help='folder containing date')
parser.add_argument('--imagedir', dest='imagedir', action='store', default='./grayscale_maps', help='image subfolder')
parser.add_argument('--predname', dest='predname', action='store', required=True, help='synaptic polarity prediction')
parser.add_argument('--segname', dest='segname', action='store', required=True, help='segmentation file')
parser.add_argument('--syn_gtname', dest='syn_gtname', action='store', required=True, help='synpatic GT')
parser.add_argument('--seg_gtname', dest='seg_gtname', action='store', default=None, help='segmentation GT')
parser.add_argument('--inputSize_xy', dest='patchSize', action='store', default=None, help='input size in xy')
parser.add_argument('--inputSize_z', dest='patchZ', action='store', default=None, help='input size in xy')



    
if __name__=='__main__':
    
    args = parser.parse_args()
    
    #train_trial = sys.argv[1]
    #train_datadir = sys.argv[2]
    #train_imagedir = sys.argv[3]
    #train_predname = sys.argv[4]
    #train_syn_gtname = sys.argv[5]
    #train_segname = sys.argv[6]
    #pdb.set_trace()
    
    train_trial = args.train_trial
    train_datadir = args.datadir
    train_imagedir = args.imagedir
    train_predname = args.predname
    train_syn_gtname = args.syn_gtname
    train_segname = args.segname
    train_seg_gtname = args.seg_gtname
    
    #patchSize = 192#428#316#204
    patchSize = int(args.patchSize)#428#316#204
    patchSize_out = 116#340#228#116
    #patchZ = 22
    patchZ = int(args.patchZ)
    patchZ_out = 4

    #pdb.set_trace()
    
    gen_data = GenerateData(train_trial,train_datadir,train_imagedir,train_predname,train_segname,  syn_gtname=train_syn_gtname,seg_gtname= train_seg_gtname, patchSize=patchSize,patchZ=patchZ)

    # def get_test_sample(self, nsamples_patch=5, nsamples_block=10, patchSize=572, patchSize_out=388, patchZ=7, patchZ_out=1, doAugmentation=False):

        
 	#pdb.set_trace()       
        
        #rangex1 = set(range(self.patchSize/2, self.grayImages.shape[2]-self.patchSize/2, patchSize_out))
        #rangex1.add(self.grayImages.shape[2]-patchSize/2)
        #rangex = sorted(list(rangex1))

        #rangey1 = set(range(patchSize/2, self.grayImages.shape[1]-patchSize/2, patchSize_out))
        #rangey1.add(self.grayImages.shape[1]-patchSize/2)
        #rangey = sorted(list(rangey1))
        
        #rangez1 = set(range(patchZ/2, self.grayImages.shape[0]-patchZ/2, patchZ_out))
        #rangez1.add(self.grayImages.shape[0]-patchZ/2)
        #rangez = sorted(list(rangez1))

	#nsamples = len(rangez)*len(rangey)*len(rangex) 
	#grayImg_set = np.zeros((nsamples, patchZ, patchSize, patchSize))
       
        #count=0
        #for z_index in rangez:
            #for y_index in rangey:
                #for x_index in  rangex:
                    #grayImg_set[count,:,:,:] = self.grayImages[z_index-(patchZ/2):z_index+(patchZ/2), y_index-(patchSize/2):y_index+(patchSize/2), x_index-(patchSize/2):x_index+(patchSize/2)]
                    
                    #count =count+1
                    
        #grayImg_set = grayImg_set[:count,...]
        #return grayImg_set
                    
    # def compute_3d_result(self, prediction1D, savename, nsamples_patch=5, nsamples_block=10, patchSize=572, patchSize_out=388, patchZ=7, patchZ_out=1, doAugmentation=False):

        #cropSize = (patchSize - patchSize_out)/2
        #csZ = (patchZ - patchZ_out)/2


        #result_vol = np.zeros(self.grayImages.shape)

        #count=0
        
        #rangex1 = set(range(patchSize/2, self.grayImages.shape[2]-patchSize/2, patchSize_out))
        #rangex1.add(self.grayImages.shape[2]-patchSize/2)
        #rangex = sorted(list(rangex1))

        #rangey1 = set(range(patchSize/2, self.grayImages.shape[1]-patchSize/2, patchSize_out))
        #rangey1.add(self.grayImages.shape[1]-patchSize/2)
        #rangey = sorted(list(rangey1))
        
        #rangez1 = set(range(patchZ/2, self.grayImages.shape[0]-patchZ/2, patchZ_out))
        #rangez1.add(self.grayImages.shape[0]-patchZ/2)
        #rangez = sorted(list(rangez1))
        
        #for z_index in rangez:
            #for y_index in rangey:
                #for x_index in  rangex:
                    
                    
                    #result_reshaped = prediction1D[count,...] 
                    #result_reshaped = result_reshaped.reshape((patchZ_out, patchSize_out, patchSize_out))
                    
                    #lbl_startz = z_index-(patchZ/2)+csZ
                    #lbl_endz = lbl_startz +patchZ_out
                    #lbl_starty = y_index-(patchSize/2)+cropSize
                    #lbl_endy = lbl_starty + patchSize_out
                    #lbl_startx = x_index-(patchSize/2)+cropSize
                    #lbl_endx = lbl_startx+patchSize_out
                    
                    #result_vol[lbl_startz:lbl_endz, lbl_starty:lbl_endy, lbl_startx:lbl_endx] =  result_reshaped
                    
                    #count =count+1
                    
        #fidw=h5py.File(savename,'w')
        #fidw.create_dataset('stack',data=result_vol.astype(np.float32))
        #fidw.close()

