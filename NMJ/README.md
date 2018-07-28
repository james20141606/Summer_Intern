
# Automatically skeletonize and segmentation


Since NMJ project contains a very large volume EM data which has some serious problems to process it automatically(hard to align, image quality is not good, axons travel fast). The project progress seems really slow. There are about 200 NMJs, and we should generate about 200 masks, each mask may contain 300 sections. So the manually seeding and segment work seems really challenging and time-consuming. I am considering to do it more automatically.

# Pipeline
The complete pipeline should contain: 
**Generating Masks —> Seeding —> Predict Membrane —> Expand Seeds —> Merge different Masks**

We would like to build up the whole pipeline, prepare all the codes and model for prediction and processing and write down the protocol.

## Predict Membrane
The automatically prediction parts must include membrane prediction, because it is “easier” to predict since the raw image already have the membrane.

##  Automatically seeding
The traditional way is to manually put seeds on each axon, but we have approximately 50,000 sections if all masks are generated, it is so time-consuming to manually put seeds. I will g**enerate seeds by distance transformation from membrane**

Then the seeds must be indexed to track each seed is from which axon, so we will manually put seeds  per 100 sections, then do **Hungarian matching.**

- Merge masks
We are thinking about linear interpolation to merge anchor sections for loop problems.

# Algorithm
## Predict Membrane
Use 3D U-net using contours from dense segmentation sections. Use 50 sections for training, then predict more, proofread predicted sections to generate more training samples. **The iterative training and predicting method will make the model more precise.**
## Automatically seeding
- Distance transformation
- Hungarian matching

This repository is a re-implementation of [Synapse-unet](https://github.com/zudi-lin/synapse-unet) (in Keras) for synaptic clefts detection in electron microscopy (EM) images using PyTorch. However, it contains some enhancements of the original model:

* Add residual blocks to the orginal unet.
* Change concatenation to summation in the expansion path.
* Support training and testing on multi-GPUs.

----------------------------

## Installation

* Clone this repository : `git clone --recursive https://github.com/zudi-lin/synapse_pytorch.git`
* Download and install [Anaconda](https://www.anaconda.com/download/) (Python 3.6 version).
* Create a conda environment :  `conda env create -f synapse_pytorch/envs/py3_pytorch.yml`

## Dataset

Use contours of Dense segmentation labels

## Training

### Command

* Activate previously created conda environment : `source activate ins-seg-pytorch`.
* Run `train.py`.

```
usage: train.py [-h] [-t TRAIN] [-dn IMG_NAME] [-ln SEG_NAME] [-o OUTPUT]
                [-mi MODEL_INPUT] [-ft FINETUNE] [-pm PRE_MODEL] [-lr LR]
                [--volume-total VOLUME_TOTAL] [--volume-save VOLUME_SAVE]
                [-g NUM_GPU] [-c NUM_CPU] [-b BATCH_SIZE]

Training Synapse Detection Model

optional arguments:
  -h, --help                Show this help message and exit
  -t, --train               Input folder
  -dn, --img-name           Image data path
  -ln, --seg-name           Ground-truth label path
  -o, --output              Output path
  -mi, --model-input        I/O size of deep network
  -ft, --finetune           Fine-tune on previous model [Default: False]
  -pm, --pre-model          Pre-trained model path
  -lr                       Learning rate [Default: 0.0001]
  --volume-total            Total number of iterations
  --volume-save             Number of iterations to save
  -g, --num-gpu             Number of GPUs
  -c, --num-cpu             Number of CPUs
  -b, --batch-size          Batch size
```

The script supports training on datasets from multiple directories. Please make sure that the input dimension is in *zyx*.

### Visulazation
* Visualize the training loss using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).
* Use TensorBoard with `tensorboard --logdir runs`  (needs to install TensorFlow).

## Prediction

* Run `test.py`.

```
usage: test.py  [-h] [-t TRAIN] [-dn IMG_NAME] [-o OUTPUT] [-mi MODEL_INPUT]
                [-g NUM_GPU] [-c NUM_CPU] [-b BATCH_SIZE] [-m MODEL]

Testing Synapse Detection Model

optional arguments:
  -h, --help                Show this help message and exit
  -t, --train               Input folder
  -dn, --img-name           Image data path
  -o, --output              Output path
  -mi, --model-input        I/O size of deep network
  -g, --num-gpu             Number of GPUs
  -c, --num-cpu             Number of CPUs
  -b, --batch-size          Batch size
  -m, --model               Model path used for test
```

## Evaluation

Run `evaluation.py -p PREDICTION -g GROUND_TRUTH`.
The evaluation script will count the number of false positive and false negative pixels based on the evaluation metric from [CREMI challenge](https://cremi.org/metrics/). Synaptic clefts IDs are NOT considered in the evaluation matric. The inputs will be converted to binary masks.




