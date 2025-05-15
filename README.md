# VENturing forth: The utility of machine learning for the morphological analysis of von Economo neurons (VENs)
This is the repository of data and code from the research "VENTuring Forth: The Utility of Machine Learning for the Morphological Analysis of von Economo Neurons (VENs). (*Ivan Banovac, Oliver Bruton, Luis Mercado-Díaz, Julian Tejada & Fernando Marmolejo-Ramos*)" Here you will find the dataset with the downloaded morphological data, as well as the images of the three-dimensional reconstructions of all morphological data downloaded from neuromorpho.org. In addition, you will also find R scripts for the machine learning processing and a Python script for GAD-CAM analysis.

## Morphological data
- `data/DataNeuroMorpho_Human_pyramidal_28-11-24.csv` contains all morphological measures extracted using L-Measure software.

## Scripts
- `scripts\MachineLearningAnalysis.Rmd` is an R Markdown document containing the machine learning model and its parametrization used to estimate the variable importance of each morphological measure when they were used to characterize von Economo neurons.
- `scripts\von_economo_vgg_testing_NewVEN_2Folds_GradCAM.py` is a Python script with the Grad-CAM model used to characterize the images of von Economo neurons.
- `scripts\ArticleFigures.Rmd` is an R Markdown document which reproduces the article figures.

### Data files to reproduce article figures
- `data/HumanVsInformationDrive_data.csv` contains the responses to the survey about which are the most important morphological measures to characterize von Economo neurons, answered by experts in neuronal morphology.
- `data/AverageRanking_variableImportance.csv` contains the results of machine learning models in terms of the importance of each morphological measure to characterize von Economo neurons.
- `data/VonEconomoScaleMeasurements.csv` contains the scaled morphological measures of all VENs.
- `data/Misclassified_VEN_cells.csv` contains the scaled morphological measures of the misclassified VENs. 
- `data/cell03b_spindle4aACC_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell03b_spindle4aACC_ScaleMeasurements neuron.
- `data/cell24_VEN_rapid_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell24_VEN_rapid_ScaleMeasurements neuron.
- `data/cell27o_spindle19aFI_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell27o_spindle19aFI_ScaleMeasurements neuron.
- `data/cell14_VEN_Cox_ScaleMeasurements.csv` contains the scaled morphological measures of the misclassified cell14_VEN_Cox_ScaleMeasurements neuron.

