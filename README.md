# Data_Science_Final_Project
Data Science in the Life Sciences - Final Project

Welcome to the final project Group 3 in Data Science in the Life Sciences! This repository contains the entire code base for our project, which we have _unofficially_ named scTherapy+. 

**Inspiration**
Tumor heterogeneity represents a major challenge in the development of both general and targeted treatment of malignant cancers. Features such as the tumoral microenvironment and intertumoral differential gene expression (including evolutionary differential expression) result in innumerable effective variations in tumoral profiles, blurring the identification of viable treatment targets, especially those that aim to be patient-specific. While the original manuscript by Ianevski et al. does lay out methods for explicitly dealing with tumoral heterogeneity, we believe that they left out potentially informative dimensions of integratable data (e.g., pathway analysis and variant calling) that could not only improve confidence in tumor classification and ultimate model predictions, but also provide a deeper level of biological interpretability – something that the original manuscript lacks. To the latter point, we believe that for this type of prediction task, biological interpretation must be clear to translate any conclusions into follow-up development of clinical treatments. 

Our method seeks to improve on the scTherapy (hence the name scTherapy+), a method developed by Ianevski et al. (2024), which uses scRNA-seq data from patient tumor samples to predict optimal patient-specific drug/dose combinations for cancer treatment. In this project, we attempt to both replicate their original method and improve its performance by integrating variant-calling and comparing different modeling frameworks. We also take the method a step further by annotating the results with pathway enrichment analysis in hopes of providing biologically sound interpretability of our models. 

Concretely, our method leverages scRNA-seq data, from which we obtain differential gene expression data and identification of transcriptome-wide SNPs, along with the chemical structures of cancer drugs, to predict post-treatment cell viability (as a percentage). 

**What's this repo?**
There are 3 branches in this repository:

- Data-Preprocessing: All data pre-processing, exploratory analysis and integration for both our internal training data and external validation set. 
- Modeling: A series of machine learning models - consisting of 3 main architectures - are trained and used for prediction on external data.
- Annotation: Predictions and corresponding explanatory data are investigated for patterns

Each branch contains its own README with more details on the methods used and how to run the code. 

The original manuscript from Ianevski et al. can be found at this link: https://www.nature.com/articles/s41467-024-52980-5#Sec8 
