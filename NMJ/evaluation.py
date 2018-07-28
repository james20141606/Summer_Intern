import argparse
import numpy as np
from scipy import ndimage
import h5py

class Clefts:

    def __init__(self, test, truth):

        test_clefts = test
        truth_clefts = truth

        self.resolution=(40.0, 8.0, 8.0)
        #self.truth_clefts_invalid = (truth_clefts == 0)

        self.test_clefts_mask = (test_clefts == 0).astype(int)
        self.truth_clefts_mask = (truth_clefts == 0).astype(int)
	
        self.test_clefts_edt = ndimage.distance_transform_edt(self.test_clefts_mask, sampling=self.resolution)
        self.truth_clefts_edt = ndimage.distance_transform_edt(self.truth_clefts_mask, sampling=self.resolution)

    def count_false_positives(self, threshold = 200):

        mask1 = 1-self.test_clefts_mask
        mask2 = self.truth_clefts_edt > threshold
        #false_positives = np.logical_and(mask1, mask2).astype(int)
        false_positives = mask1 * mask2
        return np.sum(false_positives)

    def count_false_negatives(self, threshold = 200):

        mask1 = 1-self.truth_clefts_mask
        mask2 = self.test_clefts_edt > threshold
        false_negatives = mask1 * mask2
        return np.sum(false_negatives)

    # def acc_false_positives(self):

    #     mask = 1-self.test_clefts_mask
    #     false_positives = self.truth_clefts_edt[mask]
    #     stats = {
    #         'mean': np.mean(false_positives),
    #         'std': np.std(false_positives),
    #         'max': np.amax(false_positives),
    #         'count': false_positives.size,
    #         'median': np.median(false_positives)}
    #     return stats

    # def acc_false_negatives(self):

    #     mask = 1-self.truth_clefts_mask
    #     false_negatives = self.test_clefts_edt[mask]
    #     stats = {
    #         'mean': np.mean(false_negatives),
    #         'std': np.std(false_negatives),
    #         'max': np.amax(false_negatives),
    #         'count': false_negatives.size,
    #         'median': np.median(false_negatives)}
    #     return stats

def get_args():
    parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
    # I/O
    parser.add_argument('-p','--prediction',  type=str, help='prediction path')
    parser.add_argument('-g','--groundtruth', type=str, help='groundtruth path')
    args = parser.parse_args()
    return args                    

def main():
    args = get_args()

    print('0. load data')
    test = h5py.File(name=args.prediction, mode='r',  libver='latest')['main']
    test = np.array(test)[14:-14, 200:-200, 200:-200]
    test = (test*255).astype(np.uint8)
    test[test < 200] = 0
    test = (test != 0).astype(np.uint8)

    truth = h5py.File(name=args.groundtruth, mode='r',  libver='latest')['main']
    truth = np.array(truth)
    truth = truth.transpose((2,1,0))
    truth = truth[14:-14, 200:-200, 200:-200]
    truth = (truth != 0).astype(np.uint8)

    assert (test.shape == truth.shape)
    print('volume shape:', test.shape)
    print('volume dtype:', test.dtype)
    print('number of pixels:', np.prod(test.shape))

    print('1. start evaluation')

    clefts_evaluation = Clefts(test, truth)

    false_positive_count = clefts_evaluation.count_false_positives()
    false_negative_count = clefts_evaluation.count_false_negatives()

    print('Clefts')
    print('======')

    print('\tfalse positives: ' + str(false_positive_count))
    print('\tfalse negatives: ' + str(false_negative_count))

    # false_positive_stats = clefts_evaluation.acc_false_positives()
    # false_negative_stats = clefts_evaluation.acc_false_negatives()

    # print('\tdistance to ground truth: ' + str(false_positive_stats))
    # print('\tdistance to proposal    : ' + str(false_negative_stats))

if __name__ == "__main__":
    main()    
