# PU-PieAPP - Deep Photometric Image Quality Metric

The repository contains the code for the Deep High-Dynamic-Range Image Quality metric -- PU-PieAPP trained on the [Unified Photometric Image Quality (UPIQ) dataset](https://www.repository.cam.ac.uk/handle/1810/315373)

The network trained on UPIQ dataset is identical to the original [PieAPP](https://github.com/prashnani/PerceptualImageError) architecture, however extended to account for luminance.

## Description

Increasing popularity of HDR image and video content brings the need for metrics that could predict the severity of image impairments as seen on displays of different brightness levels and dynamic range. Such metrics should be trained and validated on a sufficiently large subjective image quality dataset to ensure robust performance. As the existing HDR quality datasets are limited in size, we created a Unified Photometric Image Quality dataset (UPIQ) with over 4,000 images by realigning and merging existing HDR and SDR datasets. The realigned quality scores share the same unified quality scale across all datasets. Such realignment was achieved by collecting additional cross-dataset quality comparisons and re-scaling data with a psychometric scaling method. Images in the proposed dataset are represented in absolute photometric and colorimetric units, corresponding to light emitted from a display. We use the new dataset to train a new deep photometric image quality metric (PU-PieAPP) which outperforms existing metrics on high-dynamic image quality prediction. For more information please refer to the [project page](https://www.cl.cam.ac.uk/research/rainbow/projects/upiq/).

## Usage

Clone the repository:

```
git clone https://github.com/gfxdisp/upiq.git
cd upiq
```

To get the weights - navigate to the release section and download pupieapp_weights.pt. 

To use the weights - move the weights into the head directory (with README.md).

The code runs in Python3 with Pytorch. Dependencies can be installed by running:

```
pip3 install -r requirements.txt
```

An example running the metric:

```python
import numpy as np
import imageio
import torch as pt
from models.common import PUPieAPP

# Load saved weights
saved_state_model = './pupieapp_weights.pt'
state = pt.load(saved_state_model, map_location='cpu')

# Create and load the model
net = PUPieAPP(state)

# Set to evaluation mode
net.eval();


# Path to reference and distorted iamges
path_reference_image = './example_images/sdr_ref_1.bmp'
path_test_image ='./example_images/sdr_test_1.bmp'

# Dynamic range of the images
dynamic_range = 'sdr'

# Parameters of the display model (Assuming peak and black level of a display on which LDR image is shown).
# Set to 100 and 0.5 if unsure. The parameter is not used for HDR images as these are given in luminance values.
lum_top = 100
lum_bottom = 0.5

# The quality assessment model operates on 64x64 patches sampled on a regular grid. 
# The shift specifies the window shift for sampling the patchs. The smaller the shift the more accurate the model is.
stride = 32

# Read images 
image_ref = imageio.imread(path_reference_image)
image_ref = pt.from_numpy(imageio.core.asarray(image_ref))
image_ref = image_ref.permute(2,0,1)

image_test = imageio.imread(path_test_image)
image_test = pt.from_numpy(imageio.core.asarray(image_test))
image_test = image_test.permute(2,0,1)

# Unsqueeze to create batch dimension
image_ref = image_ref.unsqueeze(0)
image_test = image_test.unsqueeze(0)

# Run the network with no gradient
with pt.no_grad():
    score = net(image_ref, image_test, im_type=dynamic_range, lum_bottom=lum_bottom, lum_top=lum_top, stride=stride)
    
print('PU-PieAPP Quality Score: ', score.item())
```

## Citing


```
@misc{Mikhailiuk2021_1, 
author = {Mikhailiuk, Aliaksei and Perez-Ortiz, Maria and Yue, Dingcheng and Suen, Wilson and Mantiuk, Rafa{\l} K.}, 
eprint={2012.10758}, 
archivePrefix={arXiv}, 
primaryClass={eess.IV}, 
title = {{Consolidated Dataset and Metrics for High-Dynamic-Range Image Quality}}, 
year = {2020} 
}
```

## Acknowledgements

This project has received funding from EPSRC research grants EP/P007902/1 and EP/R013616/1, from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement N725253 (EyeCode), and from the Marie Skłodowska-Curie grant agreement N765911 (RealVision).
