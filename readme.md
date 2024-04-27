# Introduction

## Who Are We?
1. Denise Teh Kai Xin
2. Kaung Htet Wai Yan
3. Nguyen Xuan Nam
4. Tian Zhuoyu

## Problem Statement

There is a growing need for sentiment analysis for businesses and product owners, given the growing volumes of text-based feedback data.

## Proposed Solution

Analyse the effectivenes of machine learing models on sentiment analysis, and output a model / an ensemble of models to help predict the sentiment of any given text.

## Models used

Non-Deep Learning:
*   Decision Tree [Base]
*   Random Forest
*   Support Vector Classification (SVC)
*   Naive Bayes
*   Ada Boost
*   XGBoost

Deep Learning:
*   Sequential [Base]
*   Dense + Dropout
*   Convolutional Neural Network  (CNN)
*   Biderectional Neural Network
*   LSTM with Attention Mechanism
*   Attention Model with Review Length
*   Transformer Architecture

Other Models:
*   Vader Sentiment Analyser
*   pysentimiento

## Prerequisites

1. Create a new virtual environment. Ensure you are using Python 3.10.
2. Install project dependencies with `pip install -r requirements.txt`.
2. Next, run the following:

```python
! pip install -q keras-nlp==0.9.3
! pip install -q tensorflow==2.16.1
! pip install -q keras==3.1.1
```

## How to Use The Notebook

We saved checkpoints for almost every section of this project, from preprocessing and data analysis to model training and evaluation. The code to generate these intermediate steps have been commented out for the sake of reducing the run time. If you would like to reproduce these steps, feel free to uncomment the relevant code.

# Truncate Data

We then group the data by review score, which is either 1 (positive) or -1 (negative). As the data has 6.4 million rows, we truncated the data by randomly sampling 200000 rows, stratified by review score.

# Clean Data

We convert all text to lowercase, remove non-ASCII characters and punctuation, remove all NA values and tokenizes the text.


# Stemming

We change all words into their root form, we apply this change to every single row of the cleaned data frame. We use PorterStemmer, which has the fastest performance compared to Lancaster, Snowball and lemmetization.



# Word Embedding

In this project, we use 2 different schemes of word embedding: TF-IDF and Word2Vec.

1. TF-IDF

Term Frequency - Inverse Document Frequency (TF-IDF) is a widely used statistical method in natural language processing and information retrieval. It measures how important a term is within a document relative to a collection of documents (i.e., relative to a corpus). In this context, it measures how importance is a term in a review versus the full collection of reviews.
Term frequency is how often a vocab appears in each sentence while inverse document frequency is the logarithm of how often a sentence in the collections contain those words. TF-IDF is the product of these 2 terms. We then vectorise each sentence into a n-feature vectors where each feature is a word that are in the top n highest tf-idf inside our sentence collections.

However, [scikit-learn implementation](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)  that is used in this project is slightly different but the aim remains the same.
To note, each feature in TF-IDF is corresponding to a word.
Throughout this project, whenever we use TF-IDF it will be in feature size of 500.

2. Word2Vec

Word2Vec creates vectors of the words that are distributed numerical representation of word features - these word features could comprise of words that represent the context of the individual words present in our vocabulary. I.e a continuous distribution of vocabulary in the form of word vectors for each sentence in our context.

Heuristically, if we treat each word as a vector, the closer the distance (cosine or norms) between 2 word vectors, the closer in resemblance they are. In word vectors, features do not necessarily have any interpretation, but are more like extracted information from much higher dimensions.

Word2Vec usually has 2 approaches: continuous-bag-of-word (CBOW) and skip-gram. In short CBOW is to predict the next words given a few words versus skip-gram is to predict the surrounding word for each word. In practice, from our testing, both approaches yield almost the same result for both approaches, hence throughout this project, whenever we use word2vec, it will be a skip-gram approach.

We are using vector size i.e features of 100 throughout as it allows faster training without losing much results and also for the reduced size of the vectorised dataset.

3. Differences

As we can see from both definitions, word2vec can perform better in most models in this project due to the fact that it includes the advantage of context. TF-IDF would not capture sementic similar words like "happy" and "enjoy" but word2vec can capture it through the repetition of locality.


# Transform Data

TF-IDF: We first transform the text data in the dataframe into a sparse matrix of TF-IDF features. Then This sparse matrix is converted into a dense format and into a new dataframe and the original scores were appended to it.

Word2Vec: We first retrieve the corresponding word vector from each list of tokens. We then calculates the mean of these vectors which represents the overall semantic meaning. Last, we convert the vectors into a new dataframe where each column is 1 dimension, this DataFrame is then concatenated with the original DataFrame.


# Exploratory Data Analysis

We use 2 methods to explore the data.

Box plot: The boxplot provides a comparative view of text lengths across different sentiment scores. It helps identify if there's a significant difference in text length among different categories.

Word cloud: We have two word clouds showing words with the highest positive and negative correlations with the scores i.e. the larger the word, the more it is correlated with the positive/negative score.
    
![png](CS3244%20NoteBook_files/CS3244%20NoteBook_27_0.png)
    

Negative reviews are longer on average as compared to positive reviews. The large discrepency between the mean and median indicate that the distribution of review lengths is skewed towards shorter reviews. At the same time, there are a few outliers of very long reviews that pull away from the median.
    
![png](CS3244%20NoteBook_files/CS3244%20NoteBook_29_0.png)
    


# Split Data

We then split the data into training and testing with an 80-20 ratio.

# Model Training

We explore both non-deep learning models, as well as deep-learning models. We start each section with a baseline model, ie a basic model without tuning as comparison to our other models. For both methods, we explore both TF-IDF and word2vec schemes.

To determine and compare model performance of our binary classification task of sentiment analysis, here are the metrics we used:
  * Non-Deep Learning Models: F1 Score
    * Where The formula is given as:
    \begin{align}
          F1 = 2×\frac{precision×recall}{precision+recall}
    \end{align}

    * Precision is the measure of accurate predictions
    * Recall is the measure of relevant predctions
    * F1 provides the harmonic mean of both the Precision and Recall metrics
    * It is useful in Sentiment Analysis because we would like to consider the best possible results (ie. Most True Positives and True Negatives)
    * It also considers False Positives and Negative, thus F1 score is more robust to noise in data. Since our models are trained on pop-culture data, we are more likely to face such noise. Thus the F1 score is a good metric.

  * Deep-Learning Models: AUC-PRC
    * Area under the Precision - Recall curve
    * Showcases Overall Performance over all possible classification thresholds
    * Useful for binary classification tasks (as we are doing)
    * Similar as above

We also kept track of the accuracy of the models as it was more tangible, but our main focus is the F1 Score and AUC-PRC.


# Results and Insights
    


## Non-Deep Learning Models (F1-Score)

| Model         | TF-IDF    | Word2Vec  |
|---------------|-----------|-----------|
| Decision Tree | 0.654     | 0.629     |
| Random Forest | 0.736     | 0.729     |
| Linear SVC    | 0.741     | 0.732     |
| Naive Bayes   | 0.731     | 0.589     |
| AdaBoost      | 0.738     | 0.712     |
| XGBoost       | **0.743** | **0.743** |

## Deep Learning Models (AUC-PRC)

| Model           | Tf-idf    | Word2vec  |
|-----------------|-----------|-----------|
| Sequential      | 0.865     | 0.872     |
| Dense           | 0.819     | 0.873     |
| Dense + Dropout | **0.872** | **0.877** |
| CNN             | 0.870     | 0.870     |
| Bidirectional   | 0.865     | 0.875     |

| Model              | AUC-PRC   |
|--------------------|-----------|
| Attention          | 0.907     |
| Attention + Length | **0.908** |
| Transformer        | 0.888     |

## Best Performaning Models:
### Non-Deep Learning:
  - Tf-idf: SVC: 0.741
  - W2V: XGBoost: 0.742

### Deep Learning:
  - Attention Model with Review Length: 0.908
  - Attention Model: 0.907
  - TF-idf: Dense + Dropout: 0.871
  - W2V: Dense + Dropout: 0.876

## Insights

* Characteristics of dataset - Steam Reviews:
  - Reviews of online video games
  - Sarcasm in reviews
  - Game-specific terminology
  - Temporal changes in data

* Word2Vec embeddings based models perform better on average to Tf-idf based models:
  - W2V captures semantic relationships and context similarities, whilst Tf-idf only captures word frequency
  - In more vocabulary-specific datasets like Steam reviews, semantic information is important.

* However, for non-deep learning models, Tf-idf models performed slightly better
  - In dataasets with noise, Tf-idf might perform better for traditional models such as SVM and decision tree.
  - This corroborates with our findings, especially for Naiyes Bayes and SVC

* Attention Model VS Dense + Dropout:
  - Handling of sequential information is not done in the Dense + dropout model, that is it doesn't consider the existence of sequencing in text data. Given the characteristics of the dataset, this is an important feature.
  - Attention Model includes the additional feature review length, which helps the model account for more complexity and thus provide more regularization to prevent overfitting. Also allows for dynamic focus, and consider hierarchal relationships.

* Overall Deep-Learning performed better than non-deep learning models:
  - Deep-Learning models can capture non-linear relationships between features and target labels. More semantic understanding, rather than non-deep learning models which are based more on statistical methods.


  

# Deployment

The 2 final models we decided to use are:

1. Dense + Dropout Model (W2V)

    This is relatively simple model and provides decent performance. It provides a good balance between inference time and good predictions.

    ![png](CS3244%20NoteBook_files/CS3244%20NoteBook_113_1.png)

2. Attention Model (Without review length)
    
    More robust but longer inference time. Although the attention model with review length performs slightly better, we decided that the much longer inference time was not a good trade off.

    ![png](CS3244%20NoteBook_files/CS3244%20NoteBook_134_1.png)


## Application: Twitter Recommends

We developed application using our models, specifically a tweet-based recommendation system. By gathering sentiments on the latest tweets about any game, and aggregating it's sentiment score, we then tell the user if they should play that game. Here are the steps to set it up:



Ensure you are using Python 3.10.11

1. git clone https://huggingface.co/spaces/vnnamng/Sentimentalyst
2. Go inside the folder
3. pip install -r requirements.txt
4. [Optional] There might be a need for install tenserflow-intel.
5. Set up a .env  which is your twitter account:

Example:

```
USERNAME_TWT = "username"
PASSWORD_TWT = "password"
EMAIL_TWT = "email@email.com"
PASSWORD_EMAIL_TWT = "_"
```

Password_email is not required, put as _

6. python app.py
7. Use the app as per normal

# Error Analysis

In this section, we examine errors made by the model. By doing this, we can identify patterns that the model struggle with, providing insights into the specific cases where the model fails to generalize well. We categorise these errors as false positives and false negatives.


## Dense Dropout Model Errors

### Error Breakdown


    
![png](CS3244%20NoteBook_files/CS3244%20NoteBook_173_0.png)
    


### Most Common Words


    
![png](CS3244%20NoteBook_files/CS3244%20NoteBook_175_0.png)
    

## Attention Model Errors

## Error Breakdown
    
![png](CS3244%20NoteBook_files/CS3244%20NoteBook_183_0.png)
    

### Most Common Words

    
![png](CS3244%20NoteBook_files/CS3244%20NoteBook_185_0.png)
    

## Insights

- False Positives: Both models struggle with understanding sarcasm. For example, there were users who sarcasticly reviewed the game as 'great game 10/10', while giving negative reviews. Both of our models incorrectly predict such reviews to be positive.

- False Negatives: Both models struggles less with false negatives as compared to false positives. The false negatives also seem to be largely due to data error, where the user's negative sentiment expressed in the review does not align with the positive score they submitted. Examples of this include 'worst game i ever bought, waste of money do not buy' and 'Trash'.

- Edited reviews are prone to being misclassified. For these reviews, the users could have possibly changed their opinion of the game after a while, while the content of their original review remains. This results in a mix of sentiments.

- Neutrality: Some reviews such as "Early Access Review" and do not express sentiment of the user. A prominent number of misclassifications arise from these neutral reviews.

# Other models


In this section, we explore how our models stack up against existing models.

## VaderSentiment

[VADER-Sentiment-Analysis](https://github.com/cjhutto/vaderSentiment) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media


    
![png](CS3244%20NoteBook_files/CS3244%20NoteBook_198_0.png)
    



VADER performs poorly for our task. This could be because of its rule-based nature, relying on predefined lexicons and rules to determine sentiment polarity, intensity, and sentiment modifiers. It may struggle with more nuanced or context-dependent language found in gaming reviews.

## pysentimiento
[pysentimiento](https://github.com/pysentimiento/pysentimiento) is a Python toolkit for Sentiment Analysis and Social NLP tasks. The model supports multiple languages.
    
![png](CS3244%20NoteBook_files/CS3244%20NoteBook_203_0.png)



pysentimiento performs quite well for our task. Its performance rivals our model in terms of AUC-PRC. Being a transformer model, it has a deeper understanding of semantics and context compared to rule-based systems as it can analyze the meaning of words and phrases in relation to their surrounding context. However, it has a very slow inference time.

# Next Steps

If we had more time to work on this project, here are a few things we would like to explore:

* Other pre-trained word embeddings

    * Explore other pre-trained word embedding models, such as GloVe.

* Pre-trained transformer models

    * Use pre-trained transformer models such as BERT or GPT and fine-tune it on our dataset.

* Domain specific features

    * Explore domain specific features such as user demographics, temporal information, and gameplay time and how they influence sentiment. With more contextual information, the model might struggle less with sarcasm.

* Neutral Scores

    * Rather than binary classificaiton, we realised that it might be useful to include a 'Neutral' class, as done by many sentiment analysis models. Many reviews simply do not have a clear sentiment and having a metric of how neutral a review is might decrease misclassifications and make the model more useful.

* Explainable AI

    * As our final models are deep learning models, we would like to explore methods to improve their explanability. We can use techniques such as feature importance scores, SHAP (SHapley Additive exPlanations), or LIME (Local Interpretable Model-agnostic Explanations) to understand the contribution of each input feature to the model's predictions.


