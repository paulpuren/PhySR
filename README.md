# PhySR

Physics-informed deep super-resolution of spatiotemporal data

This paper has been accepted by the *Journal of Computational Physics*. Please see the official publication here ([Link](https://www.sciencedirect.com/science/article/pii/S0021999123005338?casa_token=VkF7g2e9Q8sAAAAA:uJ04SY6xXHeAq7sTfQ4my4kLUankf9SvvMj7MQzA8Ou87wnXXhieuUCTtZRWPvf8Z7BUeukX6Aci)) and the preprint version here ([Link](https://arxiv.org/abs/2208.01462)).

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

- Make the names of the numerical data consistent with the class ```Dataset``` and their dimension numbers. For example, in ```2DGS```, the name of the first low-resolution dataset with one specific initial condition should be like ```2DGS_IC0_2x751x32x32.mat```.
- You may manually select the dataset for training across many initial states.
- ```save_path``` is for saving models, and ```fig_save_path``` aims for saving tested figures to check the performance roughly.
- The expected outputs are: 
	- the trained model under the directory of ```save_path```.  
	- the figures of comparative results and loss history under the directory of ```fig_save_path```.
	- the tested error will be printed on the screen.
	- we also save the tested results as ```output_```, including low-resolution ```lres```, high-resolution ```hres``` and the predicted dataset ```pred```.

#### Baseline models
- [MeshfreeFlowNet](https://github.com/maxjiang93/space_time_pde): please refer to this open-source code.
- Interpolation: it is provided in ```Baseline```, trilinear method for 2D GS equation and quadlinear for 3D GS system.

#### Ablation study

The ablation codes are provided in folder ```Ablation```. The setup is similar to **Implementation**.
- w/o physics loss
- w/o ConvLSTM

## Citation

Please consider citing our paper if you find our research helpful. :D

```
@article{ren2023physr,
  title={PhySR: Physics-informed Deep Super-resolution for Spatiotemporal Data},
  author={Ren, Pu and Rao, Chengping and Liu, Yang and Ma, Zihan and Wang, Qi and Wang, Jian-Xun and Sun, Hao},
  journal={Journal of Computational Physics},
  pages={112438},
  year={2023},
  publisher={Elsevier}
}
```

## License

This project is covered under the MIT License (MIT).
