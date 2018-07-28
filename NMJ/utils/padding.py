import h5py, argparse
import os
import numpy as np

def padding_reflect(data):
    margin = ((14, 14), (200, 200), (200, 200))
    newData = np.pad(data, margin, 'reflect')
    return newData

def get_args():
    parser = argparse.ArgumentParser(description='Padding with reflection')
    # I/O
    parser.add_argument('-t','--input', help='Input path')
    parser.add_argument('-o','--output',  help='Output path')  

    args = parser.parse_args()
    return args                  

def main():
    args = get_args()
    filename = args.input
    output = args.output

    data = np.array(h5py.File(filename, 'r')['main'])
    data = padding_reflect(data)

    hf = h5py.File(str(output), 'w')
    hf.create_dataset('main', data=data)
    hf.close()

if __name__ == "__main__":
    main()