# Summer_Intern
My codes in
- **Neural Muscular Junction**
- **Synapse Prediction**
- **Synaptic Partner Prediction**
- **Synapse Clustering**

project during summer intern in [Lichtman Lab](https://lichtmanlab.fas.harvard.edu/)

# NMJ
## reconstruction

![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot/animation.gif)

## Plot segment
write python animation codes
<img src="https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot_segment/output.gif" style="width: 10px;"/>
## Seeding on Masks
![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot_segment/seeding.png)

## Automatically skeletonize and segmentation


Since NMJ project contains a very large volume EM data which has some serious problems to process it automatically(hard to align, image quality is not good, axons travel fast). The project progress seems really slow. There are about 200 NMJs, and we should generate about 200 masks, each mask may contain 300 sections. So the manually seeding and segment work seems really challenging and time-consuming. I am considering to do it more automatically.

## Pipeline
The complete pipeline should contain: 
**Generating Masks —> Seeding —> Predict Membrane —> Expand Seeds —> Merge different Masks**

We would like to build up the whole pipeline, prepare all the codes and model for prediction and processing and write down the protocol.

### Predict Membrane
The automatically prediction parts must include membrane prediction, because it is “easier” to predict since the raw image already have the membrane.

#### training steps
- train loss
![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot/trainloss.png)

- visualize output during training(Use TensorboardX)

EM image

![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot/em.png)

Ground truth image

![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot/gt.png)

Predict image

![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot/predict.png)

It seems the training is quite well after only thousands batches within one hour.

###  Automatically seeding
The traditional way is to manually put seeds on each axon, but we have approximately 50,000 sections if all masks are generated, it is so time-consuming to manually put seeds. I will g**enerate seeds by distance transformation from membrane**

Then the seeds must be indexed to track each seed is from which axon, so we will manually put seeds  per 100 sections, then do **Hungarian matching.**

- Merge masks
We are thinking about linear interpolation to merge anchor sections for loop problems.

## Algorithm
### Predict Membrane
Use 3D U-net using contours from dense segmentation sections. Use 50 sections for training, then predict more, proofread predicted sections to generate more training samples. **The iterative training and predicting method will make the model more precise.**

<img src="https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/focalloss.png" style="width: 2px;"/>

<img src="https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/focaldiceloss.png" style="width: 2px;"/>


### Automatically seeding
- Distance transformation
- Hungarian matching


# Synapse Prediction
## data augmentation
- simple augmentation
![](https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/simple.png)
- elastic augmentation
![](https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/elastic.png)
- intensity augmentation
![](https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/intensity.png)
- defect augmentation
![](https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/defect.png)

## newtwork architecture improvement
- Dink-net Dilation
- **Loss**: P: predict result, GT: ground truth, N: batch size

<img src="https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/loss.png" style="width: 2px;"/>

<img src="https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/focalloss.png" style="width: 2px;"/>

<img src="https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/focaldiceloss.png" style="width: 2px;"/>

DICE_BCE loss, not fully trained.
![DICE_BCE](https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/trainloss.png)

- network visualization
<img src="https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/Digraph.gv-1.png" style="width: 2px;"/>

# Synaptic Partner Prediction 
## metrics study
## 3D U-net & 3D CNN

# Synapse Clustering
## visualize cerebellum data
## align
## find orientation and rotate
## extract feature
## deep learning based clustering

