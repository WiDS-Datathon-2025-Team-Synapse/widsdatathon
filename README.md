# 🌎 WiDS Datathon 2025
Unraveling the Mysteries of the Female Brain: Sex Patterns in ADHD

[Link](https://www.kaggle.com/competitions/widsdatathon2025/overview) to competition on Kaggle

## 🧠 Meet Team Synapse:
- [Maya Patel](https://github.com/mpate154)
- [Julia Gu](https://github.com/juliag-27)
- [Kayla DePalma](https://github.com/kdepalma5)
- [Jannatul Nayeem](https://github.com/jannatulnayeem964)
<br/>

## 💡Project Overview
As fellows in the Break Through Tech AI Program, we participated in the WiDS Datathon 2025 on Kaggle. The WiDS Datathon Global Challenge was developed in partnership with the Ann S. Bowers Women’s Brain Health Initiative (WBHI), Cornell University, and UC Santa Barbara. The datasets and support are provided by the Healthy Brain Network (HBN), the signature scientific initiative of the Child Mind Institute, and the Reproducible Brain Charts project (RBC). This challenge provides a valuable opportunity to strengthen our data science skills while tackling an interesting and critical social impact challenge!

### *Objective*
The goal of the competition is to develop a predictive model that accurately predicts both an individual’s sex and their ADHD diagnosis using functional brain imaging data of children and adolescents and their socio-demographic, emotions, and parenting information. The challenge lies in handling complex, potentially imbalanced datasets and extracting meaningful patterns that improve prediction accuracy.

## **🎯 Project Highlights**

* Built a Graph Convolution Network (GCN) using an adjacency matrix with a half-connected graph based on Functional Connectome Matrix data to predict ADHD Outcome and Sex labels. 
* Achieved an F1 score of 0.72 and a ranking of 206 on the final Kaggle Leaderboard
* Used summations of the models guesses in each label to interpret model decisions
* Implemented device-aware computation using torch.device and reduced graph connectivity to lower computational overhead, while leveraging NumPy arrays for efficient data handling within Google Colab's hardware constraints.

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

## **👩🏽‍💻 Setup & Execution**

1. For this project, we primarily relied on Google Colab. Instructions to reproduce this data are below.
2. Download the .ipynb notebook 
3. Open in the preferred editor (VSCode, Google Colab)
4. Select the T4 GPU option (in Colab, under Runtime -> Change Runtime Type)
5. Run the uploader, which uploads the data from a public Google Drive link 
6. Run all cells that include necessary installations and imports
7. Most constructions of models will have print statements upon completion indicating the accuracy, train/test score, and F1 score. 
8. Export data using cells at the end of the file 

## **📊 Data Exploration**

Training datasets provided by the competition were named:
TRAIN_CATEGORICAL_METADATA.xlsx
TRAIN_QUANTITATIVE_METADATA.xlsx
TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv

To conduct data exploration we:
Used statistical methods such as .info() and .describe()
Constructed histograms, bar plots, cross tab plots, and count plots

During exploratory data analysis, we discovered:
More training data samples were classified as having ADHD than not (831 with ADHD and 382 without)
There were more males than females in the training data (797 males and 416 females)
There were no clear trends associated with ADHD and gender with the categorical features
Connectivity between nodes (the 200 brain regions) ranges from -0.2 to 0.4

## **🧠 Model Development**

We used different models depending on which of the three datasets we worked with, listed below.

Quantitative 
* Decision Trees
* Gradient Boosted Classifiers 
* Logistic Regressors

Categorical 
* Decision Trees
* Gradient Boosted Classifiers
Functional Connectome Matrices
* Graph Convolutional Networks 

Key model development techniques also varied widely across the three datasets. They included:

Oversampling, Undersampling, Hybrid Sampling
Model Selection
One Hot Encoding
Feature selection based on exploratory data analysis
Grid Search
Optuna
Ensemble methods
Multi-Input Models

A variety of tools and libraries were implemented for these techniques, specifically,
Torch Geometric 
Pandas
Numpy
Matplotlib
Seaborn
Optuna 
Scikit-Learn
---

## **📈 Results & Key Findings**

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

Evaluation metrics primarily focused on F1 Score and Accuracy. Some models were also analyzed for performance on a test dataset (a subset of training data that is not used in training).

Our output dataset performed within the top 200 submissions in the WiDS Datathon, with an F1 score of 0.72. On the training set, our ADHD models had accuracy of 84.5% and train/test/split accuracy of 67.5%. On the other hand, the Sex_f models had an accuracy of 83.02% and a train/test/split accuracy of 62.3%.


### *Real World Significance*
ADHD has historically been underdiagnosed and understudied in females, contributing to gaps in care and support. It affects approximately 11% of adolescents, with around 14% of boys and 8% of girls receiving a diagnosis. However, evidence suggests that girls with ADHD are often overlooked because they tend to exhibit more inattentive symptoms, which are harder to detect. As a result, undiagnosed girls may continue to struggle with symptoms that burden their mental health and daily functioning. Machine learning models capable of predicting ADHD and identifying gender-specific patterns could improve early detection, particularly in females, where diagnosis is more challenging. Additionally, these models can provide insights into the brain mechanisms underlying ADHD in both males and females, paving the way for more targeted and effective personalized treatments. Early identification and tailored therapies could significantly enhance mental health outcomes for individuals with ADHD.

<br/>

## 🔍 Data Exploration
Training Datasets:
- TRAIN_CATEGORICAL_METADATA.xlsx
- TRAIN_QUANTITATIVE_METADATA.xlsx
- TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv
