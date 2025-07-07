# AI-Sleep

AISleep: Automated and interpretable sleep staging from single-channel EEG data

Sleep EEG analysis

## Dependencies
 * Python 3.8+
 * lspopt==1.3.0 
 * matplotlib==3.5.0 
 * mne==1.6.0 
 * numpy==1.26.2 
 * pandas==2.1.3 
 * scikit-learn==1.3.2 
 * scipy==1.11.4
 * seaborn==0.13.0
 * scikit-image==0.22.0
 * statsmodels==0.14.0
 * umap-learn==0.5.5
 * yasa==0.6.3

## Embeddings Data
To reproduce the results of the paper,
we have provided the preprocessed UMAP embeddings of the Sleep-EDF-78 dataset used in the study.

The embeddings of the first subject (SC4001E0) are already available in ./data.

All embeddings can download the preprocessed UMAP embeddings data from the following link:

* [data link](https://zenodo.org/records/10217280)


## AISleep algorithm process visualization and analysis:

    ./plot_code/plot_fig.py
   
   This code reproduces figures for UMAP embedding Figure.

