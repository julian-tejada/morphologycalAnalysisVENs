# VENturing forth: The utility of machine learning for the morphological analysis of von Economo neurons (VENs)
This is the repository of data and code from the research "VENTuring Forth: The Utility of Machine Learning for the Morphological Analysis of von Economo Neurons (VENs). (*Ivan Banovac, Oliver Bruton, Luis Mercado-Díaz, Julian Tejada & Fernando Marmolejo-Ramos*)" Here you will find the dataset with the downloaded morphological data, as well as the images of the three-dimensional reconstructions of all morphological data downloaded from neuromorpho.org. In addition, you will also find R scripts for the machine learning processing and a Python script for GAD-CAM analysis.

## Morphological data
- `data/DataNeuroMorpho_Human_pyramidal_28-11-24.csv` contains all morphological measures extracted using L-Measure software.

## Scripts
- `scripts\MachineLearningAnalysis.Rmd` is an R Markdown document containing the machine learning model and its parametrization used to estimate the variable importance of each morphological measure when they were used to characterize von Economo neurons.
- `scripts\von_economo_vgg_testing_NewVEN_2Folds_GradCAM.py` is a Python script with the Grad-CAM model used to characterize the images of von Economo neurons.
- `scripts\ArticleFigures.Rmd` is an R Markdown document which reproduces the article figures.
## Analysis resutls

### Data files to reproduce article figures
- `data/HumanVsInformationDrive_data.csv` contains the responses to the survey about which are the most important morphological measures to characterize von Economo neurons, answered by experts in neuronal morphology.
- `data/AverageRanking_variableImportance.csv` contains the results of machine learning models in terms of the importance of each morphological measure to characterize von Economo neurons.
- `data/VonEconomoScaleMeasurements.csv` contains the scaled morphological measures of all VENs.
- `data/Misclassified_VEN_cells.csv` contains the scaled morphological measures of the misclassified VENs. 
- `data/cell03b_spindle4aACC_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell03b_spindle4aACC_ScaleMeasurements neuron.
- `data/cell24_VEN_rapid_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell24_VEN_rapid_ScaleMeasurements neuron.
- `data/cell27o_spindle19aFI_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell27o_spindle19aFI_ScaleMeasurements neuron.
- `data/cell14_VEN_Cox_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell14_VEN_Cox_ScaleMeasurements neuron.

### Data files containing the Grad-CAM analysis results

The files from the Grad-CAM analysis results are available via a OneDrive link due to their large size.
- `https://uconn-my.sharepoint.com/:f:/g/personal/luis_mercado_diaz_uconn_edu/EonmibSWBjlKpL-x_k2nTv0BX5s6RoT39VVZOLqz766MNw?e=4FHG6L` contains the image original screenshots of the neuronal morphologies from Neuromorpho.org used to training the model.
- `https://uconn-my.sharepoint.com/:f:/g/personal/luis_mercado_diaz_uconn_edu/EpU7rjqCmI5Gg86ag4CQXDMB9y184CYAsKvZHeTcJtQD_A?e=Mh1gQn` contains the results obtained after running the Python script  `scripts\von_economo_vgg_testing_NewVEN_2Folds_GradCAM.py`. Each subfolder calle `gradcam_result_fold*`  contains a split of the dataset used for model evaluation during a cross-validation process. A comparison of the graphical results can be observed by opening the files starting with the word comparison, such as in the next image:
   ![Comparison of the 12-2-10 model.](data/images/121-2-10.png)
   
![Uploading comparison_121-2-10.png…]()
