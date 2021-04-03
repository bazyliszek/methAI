# methAI
Containes python scripts for methAI paper

# Content

File [nn.py](https://github.com/bazyliszek/methAI/blob/main/nn.py) contains code to build and train a neural net on numerical input.
File [neuralPCAClassifier.py](https://github.com/bazyliszek/methAI/blob/main/neuralPCAClassifier.py) uses 'nn.py' to build a neural net with a specified architecture and train it to classify a data set.

# Abstract

Aberrant methylation patterns in human DNA have great potential for the discovery of novel diagnostic and disease progression biomarkers. In this paper, we used machine learning algorithms to identify promising methylation sites for diagnosing cancerous tissue and to classify patients based on methylation values at these sites.
We used genome-wide DNA methylation patterns from both cancerous and normal tissue samples, obtained from the Genomic Data Commons consortium and trialled our methods on three types of urological cancer. A decision tree was used to identify the methylation sites most useful for diagnosis.
	The identified locations were then used to train a neural network to classify samples as either cancerous or non-cancerous. Using this two-step approach we found strong indicative biomarker panels for each of the three cancer types.
These methods could likely be translated to other cancers and improved by using non-invasive liquid methods such as blood instead of biopsy tissue.

# Keywords
AI, deep learning, biomarkers, decision tree, DNA methylation, machine learning, neural network, GDC, TCGA, urological cancers, prostate, kidney, bladder


# Simplified pipeline
![alt text](https://github.com/bazyliszek/methAI/blob/main/img/basic%20pipeline.png)

# Using the tool

# Requirements

Python 3.0

# Data availability
All example datasets were used in previously published studies. We used publicly available and anonymous data from The Cancer Genome Atlas and Harmonized Cancer Datasets in the Genomic Data Commons Data Portal (https://www.cancer.gov/tcga , https://portal.gdc.cancer.gov/) investigating kidney, prostate and bladder tissues (dataset https://portal.gdc.cancer.gov/legacyarchive/search/f) assayed in multiple, independent experiments using the Illumina 450k microarray. The clinical metadata and the manifest file of the samples were downloaded through the GDC legacy archive on 2019-10-19 and 2019-11-09.  We used the Genomic Data Commons (GDC) data download tool (https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) using the manifest files deposited on our GitHub.
All methylation data were already normalized across samples by the GDC consortium and therefore were ready for further processing. Nevertheless, to further confirm that the normalization between datasets was done correctly by GDC we plotted a histogram of the two datasets for each tissue type and calculated the first four moments for the distributions between cancer and normal samples for each tissue type (Supplementary S1) to check that there were no noticeable differences between the two groups. 
