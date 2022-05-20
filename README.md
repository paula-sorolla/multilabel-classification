# Multilabel classification using BERT embeddings

This is the main repository for the Internship Thesis developed by Paula Sorolla.
The project is intended to provide improvements and new approaches to the document-level classification component in the Canary pipeline.

The goal and all previous work made at Elsevier are summarized in the following [Confluence page](https://confluence.elsevier.com/display/ContentAssets/Document-Level+Classification).

# Table of Contents
1. [Repository Structure](#repository-structure)
2. [Setup](#Setup)
3. [Network architecture](#network-architecture)




# Repository Structure
Below is the structure of the repo with a brief description.
```
repo/
  notebooks/    BERT_classifier.ipynb - Includes the working notebook where the first experiments have been setup and run
  src/
    ClassifierModels.py - Includes the Classifier network models to be chosen for training in the experimental phase
    LossFunctions.py - Includes different Loss functions definitions to be chosen for training in the experimental phase
    Postprocessing.py - Includes functions to compute metrics and retrieve results form the experiments
    Preprocessing.py - Includes functions to load the and setup the required datasets necessary for the experiments
    utils.py - Help functions used throughout the process
  Inference.py - Includes the Inference metohds for testing the multilabel classification
  main.py - Main executable file for the train/testing of the model
  README.md - README providing useful information of the project
  requirements_conda.txt - The requirements file for reproducing the analysis in an Anaconda environment
  requirements.txt - The requirements file for reproducing the analysis in an Python3 environment
  Train.py - Includes the Training and Evaluation metohds for the multilabel classification
  
```

# Setup



# Network architecture

This repository includes 2 different neural network arquitectures that are being currently tested for comparison:

![BERT base classifier](./img/BERTbase.png)
1. BERT base classifier using SciBERT followed by a 


![BERT LSTM classifier](./img/BERTlstm.png)

<!-- ## Setup



`export PY_ARTIFACTORY_TOKEN=`

```bash
    make docker_update_ci
  ``` -->