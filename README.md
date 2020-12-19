# nlp-product-sentiment-classification

<img class="irc_mi" src="https://inudgeyou.com/wp-content/uploads/2019/03/p%C3%A6re1.jpg" data-atf="0" width="400" height="300" style=""/></a>
<img class="irc_mi" src="https://github.com/Iskriyana/nlp-product-sentiment-classification/blob/master/notebooks/01_exploration/text_star.png" data-atf="0" width="400" height="300" style=""/></a>

## -- Project Status: done

## Project Intro
The purpose of this project is to develop a model to classify various products into 4 different classes of sentiments based on the raw text description.


### Collaborators
|Name               |  Github Page                              |  Personal Website  |
|-------------------|-------------------------------------------|--------------------|
|Iskriyana Vasileva | [iskriyana](https://github.com/Iskriyana) |

### Methods Used
* small data set --> cross-validation 
* imbalanced data set --> oversampling with SMOTE
* text and non-text data --> multi-input neural network
* use of pre-trained word embeddings

### Technologies
* Python
* Jupyter Lab
* PyCharm

## Project Description
* The project is based on this Kaggle's dataset: https://www.kaggle.com/akash14/product-sentiment-classification


## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. `conda create -n nlp-sent python=3.6`
3. `conda activate nlp-sent`
4. `pip install -r requirements.txt`
5. (wip) Raw Data is being kept [here](Repo folder containing raw data) within this repo.
6. (wip) Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
7. (wip) Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables
* [01_1_Data_Exploration](#notebooks/01_exploration/01_1_Data_Exploration.ipynb)
* [02_1_Text_Preprocessing_with_TF](#notebooks/02_processing/02_1_Text_Preprocessing_with_TF.ipynb)
* [02_2_NLP_Model_Choice_Optimisation](#notebooks/02_processing/02_2_NLP_Model_Choice_Optimisation.ipynb)
* [02_2_Product_Type_Model](#notebooks/02_processing/02_2_Product_Type_Model.ipynb)
* [02_3_Multi_Input_Model](#notebooks/02_processing/02_3_Multi_Input_Model.ipynb)
* wip [Blog Post](#)

## Future Work or Actions to Further Improve the Model
* more data
* more hyperparameter tunning  - hidden units, batch size etc. 
* test the model performance without very specific words such as  'sxsw'
* try without one-hot encoding of the labels
* migrate from notebooks to scripts
* use optimal epoch
---

This file structure is based on the [DSSG machine learning pipeline](https://github.com/dssg/hitchhikers-guide/tree/master/sources/curriculum/0_before_you_start/pipelines-and-project-workflow).
