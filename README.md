# Text Classification of Amazon Food Reviews
![](/images/git_splash.jpg)

#### --Project Status: In progress

## Project Intro/Objective
The purpose of this project is to use labeled text reviews to predict unlabeled text reviews. Learning from review examples that have ratings out of five, the model then makes predictions of five star scores based on newly seen text.

### Methods Used
* NLP Techniques
* Word embeddings
* Convolutional Neural Networks
* Data Visualization
* Predictive Modeling

### Packages/Dependencies
* Python >= 3.6
* jupyter notebooks
* Keras >= 2.3.0
* Pandas 
* scikit-learn >= 0.22.1
* matplotlib >= 3.0.1

## Project Description
This project is based off of a dataset used for a [Stanford paper](https://arxiv.org/abs/1303.4402) on modeling how user reviews change over time. Compared to other review datasets, I found this one attractive because of its emphasis on users with multiple reviews over many years, thereby guaranteeing a greater review quality versus reviews from bot accounts. The exact dataset is most easily accessible [here on kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews). There is also an accompanying command line app, in which users put in their own reviews and are returned predicted scores.

## Project Contents
|Name     |  Description   | 
|---------|-----------------|
|[main.ipynb](/main.ipynb)| The notebook that contains the main analysis and modeling for the project as well the previous iteration of the modelling approach.|
|[library.py](/library.py) | The library containing code for handling certain repeated text transformations and finer metrics for Keras. There is also library code for unused TF-IDF vectorization of words however it was too computationally complex for the size of this dataset and was less effective on aggregate than the custom embedding method.|
|[visualisation.py](/visualisation.py)| A library for visualising Keras model performance.|
|[slidedeck.pdf](/slidedeck.pdf)| A slide presentation for those interested in the project; presented at Flatiron 23/01/20|
|[conv_embedding_model_lrg.h5](/conv_embedding_model_lrg.h5)| The saved model for use in predictive capacity such as in the app accompanying this project (link TBC)|
|[tk_20k_vocab_200_words.pkl](/tk_20k_vocab_200_words.pkl)| A pickle of the fitted tokenizer used in the main notebook, which can be reused to tranform new documents with the same parameters. |

## Results
The final model has 66% predictive accuracy across five classes on unseen data. The times it gets it wrong it will almost always predict and adjacent class.

## The App
Accompanying this project is a command app built off of the saved components of this project that takes any text review and score that a user inputs and tell them their predicted score. The app records the text, their score and the predicted score in a CSV
![](/images/app_example.jpg)

## Usage
Please feel free to clone or fork this repo if you fancy and get in touch if you are at all interested in this project.


