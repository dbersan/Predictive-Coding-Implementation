# Predictive Coding Model

This repository contrains an implementation of the paper *An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity, Whittington, James CR and Bogacz, Rafal [2017]* - [link](https://www.mitpressjournals.org/doi/full/10.1162/NECO_a_00949). 

Predictive Coding is an alternative approach to artificial neural networks, inspired by theories of brain function. The network is structured in a hierarchical fashion with multiple layers (similar to deep learning), in which bottom-most layers receive *sensory input*, while the higher layers are responsible for representing the underlying *hidden causes* of such inputs (i.e. the inferred structure of the real world which gave rise to such input, i.e. explain the input). 

Meanwhile, top-bottom (feedback) connections generate the predicted input given the hidden causes encoded in the higher layers, which are then compared with the actual input. Finally, their difference (the prediction error) is transmitted through bottom-up (forward) connections, adjusting the actual valued of the hidden causes such that they better explain the sensory information. 

There are many flavors of the Predictive Coding framework, and one in particular is a simple classifier for a static input network (that is, the input data does not contain a temporal dimension). 

## Run Latest Model 

We compare the performance of a typical Backprop network againt the Predictive Coding implementation, on the Imagenet 64x64 dataset. 

To run these exaples, you'll need at least around 16 Gb of *free* RAM, because the whole dataset needs to fit into memory. Otherwise, you may easily select a subset of the data by modifying the code. 

1. Download the `Train(64x64) part1`, `Train(64x64) part2` and `Val(64x64)` (select `npz format`) from the [imagenet website](https://image-net.org/download-images). Extract inside `datasets/imagenet-64x64/`.

2. To run the Backpropagation network, run...

3. To run the Predictive Coding network, run...




