# Consolidated Dataset and Metrics for High-Dynamic-Range Image Quality

Increasing popularity of \ac{HDR} image and video content brings the need for metrics that could predict the severity of image impairments as seen on displays of different brightness levels and dynamic range. Such metrics should be trained and validated on a sufficiently large subjective image quality dataset to ensure robust performance. As the existing \ac{HDR} quality datasets are limited in size, we created a Unified Photometric Image Quality dataset (UPIQ) with over 4,000 images by realigning and merging existing \ac{HDR} and \ac{SDR} datasets. The realigned quality scores share the same unified quality scale across all datasets. Such realignment was achieved by collecting additional cross-dataset quality comparisons and re-scaling data with a psychometric scaling method. Images in the proposed dataset are represented in absolute photometric and colorimetric units, corresponding to light emitted from a display. We use the new dataset to retrain existing HDR metrics and show that the dataset is sufficiently large for training deep architectures. We show the utility of the dataset on brightness aware image compression.

If using the trained metrics, or the dataset, please cite:

Aliaksei Mikhailiuk, Maria Perez-Ortiz, Dingcheng Yue, Wilson Suen, Rafal K. Mantiuk. Consolidated Dataset and Metrics for High-Dynamic-Range Image Quality IEEE Transactions on Computational Imaging, (under review), 2020 

## Repository overview


