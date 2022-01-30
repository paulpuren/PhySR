# PhySR

Physics-informed deep super-resolution of spatiotemporal data

The paper is currently under review.

## Overview

High-fidelity simulation of complex physical systems is exorbitantly expensive and inaccessible across spatiotemporal scales. Therefore, we propose a new approach to augment scientific data with high resolution based on the coarse-grained data by leveraging deep learning. It is an efficient spatiotemporal super-resolution (ST-SR) framework via physics-informed learning. Inspired by the independence between temporal and spatial derivatives in partial differential equations (PDEs), this method decomposes the holistic ST-SR into temporal upsampling and spatial reconstruction regularized by available physics. Moreover, we consider hard encoding of boundary conditions into the network to improve solution accuracy. Results demonstrate the superior effectiveness and generalizability of the proposed method compared with baseline algorithms through numerical experiments. This repo is for our ``PhySR`` model, including the source code and the numerical datasets we have tested. The experimental test may be added in the future.

## System Requirements

### Hardware requirements

We train our ``PhySR`` and the baseline models on an Nvidia DGX with four Tesla V100 GPU of 32 GB memory. 

### Software requirements

#### OS requirements
 
 - Window 10 Pro
 - Linux: Ubuntu 18.04.3 LTS

#### Python requirements

- Python 3.6.13
- [Pytorch](https://pytorch.org/) 1.6.0
- Numpy 1.16.5
- Matplotlib 3.2.2
- scipy 1.3.1

## Installtion guide

It is recommended to install Python from Anaconda with GPU support, and then install the related packages via conda setting.  

## How to run

### Dataset

Considering the traing data size being over large, we provide the ```code``` for data generation used in this paper, including 2D and 3D Gray-Scott reaction-diffusion equations. They are coded in the high-order finite difference method. The initial conditions (ICs) are manually made by adding several initial disturbance at different locations. 

### Implementation

We show the PhySR for 2D and 3D Gray-Scott reaction-diffusion equations in the folder ```Code```. 




## License

This project is covered under the MIT License (MIT).
