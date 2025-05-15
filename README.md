# VENturing forth: The utility of machine learning for the morphological analysis of von Economo neurons (VENs)
This is the repository of data and code from the research "VENTuring Forth: The Utility of Machine Learning for the Morphological Analysis of von Economo Neurons (VENs)." Here you will find the dataset with the downloaded morphological data, as well as the images of the three-dimensional reconstructions of all morphological data downloaded from neuromorpho.org. In addition, you will also find R scripts for the machine learning processing and a Python script for GAD-CAM analysis.

## Morphological data
- `data/DataNeuroMorpho_Human_pyramidal_28-11-24.csv` contains all morphological measures extracted using l-measure software.
- `data/images` contains screenshots of all sample neurons in wich in `original` are screenshots of the neuronal morphology from Neuromorpho.org; in `withSoma` are screenshots of the SWC file using the HBP Neuron Morphology online viewer (Velasco et al. 2024); and `withDiameter` are screenshots of the SWC file using the NEURON software 3D import tool (Hines and Carnevale 2001) with the “show diameter” option checked.

## Scripts
- `scripts\MachineLearningAnalysis.Rmd` is a R markdonw with the machine learning model and their parametrization used to estimated variable importance of each morphological measures when their were used to characterize von Economo neurons
- `scripts\von_economo_vgg_testing_NewVEN_2Folds_GradCAM.py` is a Python script with the GradCAM model used to characterize the images from von Economo neurons 
- `scripts\ArticleFigures.Rmd` is a R markdonw which reproduce the article images 

### Data files to reproduce article figures
- `data/HumanVsInformationDrive_data.csv` contains the responses to the survey about which are the most important morphological measures to characterize von Economo neurons answered by experts in neuronal morphology
- `data/AverageRanking_variableImportance.csv` contains the results of machine learning models in terms of the importance of each morphological measures to characterize von Economo neurons
- `data/VonEconomoScaleMeasurements.csv` contains the scaled morphological measures of all VENs
- `data/Misclassified_VEN_cells.csv` contains the scaled morphological measures of the misclassified VENs 
- `data/cell03b_spindle4aACC_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell03b_spindle4aACC_ScaleMeasurements neuron
- `data/cell24_VEN_rapid_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell24_VEN_rapid_ScaleMeasurements neuron
- `data/cell27o_spindle19aFI_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell27o_spindle19aFI_ScaleMeasurements neuron
- `data/cell14_VEN_Cox_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell14_VEN_Cox_ScaleMeasurements neuron

