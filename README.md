# Product Sentiment Classification (NLP)

<img class="irc_mi" src="https://github.com/Iskriyana/nlp-product-sentiment-classification/blob/master/notebooks/01_exploration/text_star.png" data-atf="0" width="400" height="400" align="center" style=""/>

## Project Description
* The purpose is to develop a model to classify various products into 4 different classes of sentiments based on the text description and the product type.
* Below is an example of the data where:
    * Inputs are: 
        * product_description = text part 
        * product_type = non-text part
    * The target is the sentiment
        * 0 - cannot say
        * 1 - negative
        * 2 - positive
        * 3 - no sentiment
* The project uses this [Kaggle dataset](https://www.kaggle.com/akash14/product-sentiment-classification)
        
|Product_Description |Product_Type |Sentiment|
|---|---|---|
|The Web DesignerÛªs Guide to iOS (and Android) Apps, today @mention 10 a.m! {link} #sxsw |9 |2 |
|RT @mention Line for iPad 2 is longer today than yesterday. #SXSW  // are you getting in line again today just for fun? |9 |2 |
|Crazy that Apple is opening a temporary store in Austin tomorrow to handle the rabid #sxsw eye pad too seekers          |9 |2 |


### Methods Used
The challenges of this project were: 
* Small data set - there are only 6364 data entries. This is challenging primarily for the text part, for which I wanted to use a deep learning model.
* Imbalanced data set - almost 60% of the entries are classified as 2 (positive) and 33% as 3 (did not have a sentiment). 
* Text and non-text data - the dataset at hand has two types of input data - text and non-text. As a result it requires multimodal inputs. 

The above issues were addressed as follows: 
* K-fold cross-validation (sklearn.model_selection KFold) was used as a first method to tackle the small data set. It ensured that every observation from the original dataset has the chance of appearing in the training and test set. 
* While searching for the optimal model, pre-trained embeddings ([GloVe](https://nlp.stanford.edu/projects/glove/)) were tried. This was done in an attempt to add more variability to the text part. However, due to the specific nature of the descriptions they were too general and the models trained with them did not perform better than a model with "homegrown" embeddings or bag-of-words.
* The imbalanced data was tackled with oversampling using SMOTE (Synthetic Minority Oversampling Technique). The synthetic increase of the minority classes contributed once more to tackling the small data set. 
* The two types of data were addressed by using different types of neural layers to address the "needs" of the 2 types separately. They were then united in one single model in order to "jointly learn" by seeing all input information simultaneously. In order to do this, I used the Keras (Model API). 
 
 
### Tech Stack 
* Python main libraries
    * Pandas, NumPy
    * Tensorflow, Keras
* Jupyter Lab
* PyCharm
* Github - the file structure is based on the pipeline and project workflow template of  [DSSG](https://github.com/dssg/hitchhikers-guide/tree/master/sources/curriculum/0_before_you_start/pipelines-and-project-workflow). 

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. `conda create -n nlp-sent python=3.6`
3. `conda activate nlp-sent`
4. `pip install -r requirements.txt`
5. Raw Data is being kept [here](https://github.com/Iskriyana/nlp-product-sentiment-classification/tree/master/data/01_raw) within this repo.
6. Data processing/transformation scripts are being kept [here](https://github.com/Iskriyana/nlp-product-sentiment-classification/tree/master/notebooks/02_processing)

## Featured Notebooks 
#### [Exploratory Data Analysis](https://github.com/Iskriyana/nlp-product-sentiment-classification/blob/master/notebooks/01_exploration/01_1_Data_Exploration.ipynb)
This step made clear that the data set at hand is small and imbalanced. By doing word clouds per product type, one could had a decent guess of what the products might have been (Google phone, iPad, iPhone etc.) 

<img class="irc_mi" src="https://github.com/Iskriyana/nlp-product-sentiment-classification/blob/master/notebooks/01_exploration/wordclouds_per_product_type.png" data-atf="0" width="500" height="300 " style=""/></a>

#### [Text Processing and Tokenisation with Tensorflow](https://github.com/Iskriyana/nlp-product-sentiment-classification/blob/master/notebooks/02_processing/02_1_Text_Preprocessing_with_TF.ipynb)
In this notebook the text was "normalised", i.e. stopwords were removed, contractions expanded, the text was lemmatised, special characters removed and finaly the text was turned into sequences of indexes via the Tensorflow Tokenizer. 

#### Model Development
It included 3 phases. In the first 2 the models for the text and non-text inputs were chosen and optimised accordingly. In the last 3rd part they were brought together in a multi-input neural network, using the Keras Model API 

[NLP Model](https://github.com/Iskriyana/nlp-product-sentiment-classification/blob/master/notebooks/02_processing/02_2_NLP_Model_Choice_Optimisation.ipynb)
* the goal of this notebook was to identify the best models in terms of F1 score on the training and validation data for the **text** part of the data
* The following models were evaluated: 
    * bag-of-words
    * a fully connected NN with "homegrown" embeddings layer
    * a fully connected NN with pre-trained embeddings layer
    * a "homegrown" embeddings layer with LSTM 
    * a pre-trained embeddings layer with LSTM
    * a "homegrown" embeddings layer with Conv1D 
    * a pre-trained embeddings layer with Conv1D
* the top 2 models were then regularised in order for them to generalise better on unseen data
* at the end the best performing model with regularisation was the bag-of-words with dropout

[Product Type Model](https://github.com/Iskriyana/nlp-product-sentiment-classification/blob/master/notebooks/02_processing/02_2_Product_Type_Model.ipynb) - a simple fully connected neural network was used for this part

[Multi-Input Model](https://github.com/Iskriyana/nlp-product-sentiment-classification/blob/master/notebooks/02_processing/02_3_Multi_Input_Model.ipynb) - the two models above were put together into a multi-input model

## Results
* The baseline model used was sklearn's DummyClassifier, which predicts the majority class. It achieves an F1 of 19%. 
* The main metric was the F1-score due to the imbalanced nature of the data set. 

#### Text Part (NLP)
* The bag-of-words with dropout turned out to be the best model. It achieved 64% F1 score on the test data.
* My expectations were that a model with pre-trained embeddings in combination with an LSTM or Conv1D will outperform the rest. 
* My reasoning was that the pre-trained embeddings will enforce the generalisation while an LSTM or Conv1D will help capture the text sequence. 
* However, it turned out: 
    * that for this particular data set "homegrown" embeddings are better than pre-trained. This can be explained by the small amount of data and the tech concentrated nature of the text descriptions. These "specificities" were better captured by training the data's own embeddings (that is why also "homegrown"). 
    * that bag-of-words is better than any model with embeddings. This for me was the biggest surprise. After a discussion with a mentor of mine an explanation can be that due to the short length of the text descriptions and relatively "loose" way of writing them, semantics and text structure do not play a significant role. Much more important is, if a word is present, which is what bag-of-words captures. 

#### Non-Text Part 
* The Model achieved 44% F1 on the test data 
* This is not a very good result and is to be explained mostly by the fact that there is only one feature and little data
* However, it is still much better than the baseline model
* Therefore, the model was picked to be used in the multi-input model

#### Multi-Input Model
* A F1-score of 62% was achieved on the test data.
* The results on the final test set show a room for improvement. 
* Given all the limitations, however, it is still a good enough performance (and better than the baseline)

## Next Steps to Improve the Model 
* More data - especially for the text part, more data will improve the model by adding more variability
* Adding new non-text features - additional information such as OS version, product model etc. will also enhance the performance of the non-text model
* More hyperparameter tunning  - by both models more tests can be performed in order to find the best set of hyperparameters hidden units, batch size etc. One can also take a look at automatic hyperparameter tuning techniques.  
* Test the model performance without very specific words such as  'sxsw' - among the most frequent words is the abbreviation sxsw. According to wikipedia it is "an annual conglomeration of parallel film, interactive media, and music festivals and conferences" ([wikipedia article](https://en.wikipedia.org/wiki/South_by_Southwest)). Due to the technical theme of the descriptions, I left the word. The same also for x89 (a tablet model). It would be interesting to test the impact of such specific words on the performance of the models. 
* Try without one-hot encoding of the labels - the labels are currently one-hot encoded. However, as they indicate a preference or the lack thereof, one can test how their unprocessed use would impact the model performance. 
* Use early stopping - while training, one can try the impact of stopping the training at the optimal epoch, i.e. where the loss is at its lowest. 
---


