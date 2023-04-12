# Machine learning for detecting extragalactic globular clusters

Data and code for the publication ["Evaluating the feasibility of interpretable machine learning for globular cluster detection"](https://doi.org/10.1051/0004-6361/202243354).

## Introduction

Extragalactic globular clusters (GCs) are important tracers of galaxy formation and evolution because their properties, luminosity functions, and radial distributions hold valuable information about the assembly history of their host galaxies. Obtaining GC catalogues from photometric data involves several steps which will likely become too time-consuming to perform on the large data volumes that are expected from upcoming wide-field imaging projects such as Euclid. Here, we provide a data set and code for exploring the feasibility of various machine learning methods to aid the search for GCs in extensive databases.

## Data

### Overview

We provide an extensive data set for evaluating machine learning models on the task of detecting GCs. In total, the data set consists of `84929 sources including 18556 GCs and 63829 non-GCs`. It can be further split into two data sets (e.g., for training and testing on different environments), one containing data from the `Fornax galaxy cluster (21767 sources in total including 6161 GCs and 15606 non-GCs)` and one containing data from the `Virgo galaxy cluster (63162 sources in total including 12395 GCs and 50767 non-GCs)`.
The data is provided either in the form of 20x20 pixel image cut-outs of sources or in the form of tabular data listing physically meaningful features (e.g., magnitudes, colours, etc.) extracted from the images.

Image data is provided in astropy.fits format, while tabular data is provided as csv tables. 
Data loading and preprocessing routines are included (with examples) in the code.

### Files

The data sets can be found in the `data/` folder. 
Image files are in `data/ImageData`, while the tabular data for both galaxy clusters is in `data/ACS_sources_original.csv`.
We further include data for NGC1427a for additional model testing.

### Details on data set generation

To generate the labelled data set, we use the data from the Advanced Camera for Surveys (ACS) Virgo Cluster Survey (ACSVCS; Côté et al. 2004) and ACS Fornax  Cluster  Survey  (ACSFCS;  Jordán  et  al.  2007a),  in  which GCs are marginally resolved due to the close distances of the Virgo  and  Fornax  galaxy  clusters  (16.5  Mpc  and  20  Mpc,  respectively). Sources were labelled via cross-referencing using existing GC catalogues (Jordán et al. 2007, 2015). Both the ACSVCS and the ACSFCS are surveys based on Hubble Space Telescope (HST) ACS observations in the F475W (∼g) and F850LP (∼z) bands of 100 massive early-type galaxies in Virgo and 43 galaxies in the Fornax galaxy cluster, respectively. The code used for creating the tabular data is in `data/data_extraction_routines.ipynb`.

## Code

We provide implementations, evaluation schemes and experimental setups using several machine learning models, such as logistic regression, support vector machines, k nearest neighbour classifiers, tree-based algorithms like random forests and boosted trees and neural architectures like multilayer perceptrons (MLP), convolutional neural networks (CNN) and TabNet (Arik & Pfister 2021). We further demonstrate how to use explainable artificial intelligence techniques like LIME (Ribeiro et al. 2016) to explain model predictions.

To use our code, install the provided package globsML via pip
```
pip install -e .
```
from the folder containing the `setup.py`. 
The source code is located in `globsML/`. 
Example notebooks replicating the experiments shown in the publication can be found in `experiments/`. 

## Citation

If you use the provided data set or code, or find it helpful for your own work, please cite

```
@article{dold2022evaluating,
  title={Evaluating the feasibility of interpretable machine learning for globular cluster detection},
  author={Dold, Dominik and Fahrion, Katja},
  journal={Astronomy \& Astrophysics},
  volume={663},
  pages={A81},
  year={2022},
  publisher={EDP Sciences}
}
```

## Acknowledgments

This work is based on observations made with the NASA/ESA Hubble Space Telescope, and obtained from the Hubble Legacy Archive, which is a collaboration between the Space Telescope Science Institute (STScI/NASA), the Space Telescope European Coordinating Facility (ST-ECF/ESA) and the Canadian Astronomy Data Centre (CADC/NRC/CSA). This work made use of Astropy, a community-developed core Python package for Astronomy (Astropy Collaboration et al. 2013, 2018). This work made use of Photutils, an Astropy package for detection and photometry of astronomical sources (Bradley et al. 2020). Furthermore, this work made use of the open source libraries Scikit-Learn (Pedregosa et al. 2011), PyTorch (Paszke et al. 2019), NumPy (Harris et al. 2020), Matplotlib (Hunter 2007), pandas (Reback et al. 2020; Wes McKinney 2010), tqdm (da Costa-Luis et al. 2021) and CatBoost (Prokhorenkova et al. 2018).
