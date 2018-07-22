# Summer_Intern
My codes about Synapse Prediction, Synaptic Partner Prediction and Synapse Clustering during summer intern in Lichtman Lab

# NMJ
## Plot segment
write python animation codes
<img src="https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot_segment/output.gif" style="width: 10px;"/>

# Synapse Prediction
## data augmentation
- simple augmentation
- elastic augmentation
- intensity augmentation

## newtwork architecture improvement
- Dink-net dilation
- loss: P: predict result, GT: ground truth, N: batch size
<img src="http://chart.googleapis.com/chart?cht=tx&chl= L = 1 - \frac{2 \times \sum_{i=1}^N  |P_i \cap GT_i  | }{\sum_{i=1}^N  (P_i + GT_i)} + \sum_{i=1}^N BCELoss(P_i,  GT_i)" style="border:none;">
$$L = 1 - \frac{2 \times \sum_{i=1}^N  |P_i \cap GT_i  | }{\sum_{i=1}^N  (P_i + GT_i)} + \sum_{i=1}^N BCELoss(P_i,  GT_i)$$

- network visualization
![network](https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/Digraph.gv-1.png)

# Synaptic Partner Prediction 
## metrics study
## 3D U-net & 3D CNN

# Synapse Clustering
## visualize cerebellum data
## align
## find orientation and rotate
## extract feature
## deep learning based clustering


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
