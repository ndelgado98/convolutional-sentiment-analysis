import pandas as pd
import numpy as np
from nltk.tokenize import casual_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.utils import resample
from sklearn.utils import shuffle 
from collections import Counter
from collections import OrderedDict
import string
from keras import backend as K



### NLP Preprocessing ###

'''
This part of the library is for preprocessing textual data. The entirety of this library is not used in the main notebook, but is there incase anyone should want to approach preprocessing or vectorization differently.
'''

sw = stopwords.words('english')
sw.remove('not')
sw.remove('no')


def remove_stopwords(doc: list or str, join: bool = False) -> list or str:
    """
    Removes overly frequent words that give less information about the document. Takes in either a string or list of strings and returns a string or a list of strings, depending on the ordering of text preprocessing.

    Parameters:
    doc (list or str): takes in individual documents either as full strings or list of strings in case tokenization has taken place first.

    join (bool): defines whether or not the output should be a list of tokens or a joined string. 

    Returns:
    list or str: returns either a list of tokens or joined string depending on 'join' parameter.
    """
    if not join:
        if isinstance(doc, list):
            filtered_tokens = [w for w in doc if w not in sw]
            return filtered_tokens
        elif isinstance(doc, str):
            doc = doc.split()
            filtered_tokens = [w for w in doc if w not in sw]
            return filtered_tokens
        else:
            raise TypeError("argument doc takes type lst or type str")
    elif join:
        if isinstance(doc, list):
            filtered_tokens = [w for w in doc if w not in sw]
            return " ".join(filtered_tokens)
        elif isinstance(doc, str):
            doc = doc.split()
            filtered_tokens = [w for w in doc if w not in sw]
            return filtered_tokens
        else:
            raise TypeError("argument doc takes type lst or type str")
        
    

def tf(document: str or list) -> dict:
    """
    Takes in a document and returns the relative frequencies of each word in the document.

    Parameters:
    document (str or list): the document to process

    Returns:
    dict: a dictionary with the words of the document as the keys and their relative frequencies as the values.
    """
    if isinstance(document, str):
        document = document.split()
        
    elif isinstance(document, list):
        pass
    
    else:
        raise TypeError
        
    bow = dict(Counter(document))
    for word, count in bow.items():
        bow[word] = count/len(document)
    return bow

def idf(corpus: list, verbose: bool = False) -> dict:
    """
    Calculates the inverse document frequencies for each word within a corpus. Inverse document frequency is the log of number of documents in the corpus divided by the number of documents in which the word in question appears. 

    Parameters:
    corpus (list): takes in a corpus in the form of a list of documents. Documents must be split/tokenized already. 
    verbose (bool): on large corpora, idf can take a long time, the verbose option prints out how many documents have been processed. 

    Returns:
    dict: a dictionary of with the words as keys and their idf scores as their values.
    """
    output = []
    vocab = set()
    for doc in corpus:
        for word in doc:
            vocab.add(word)

    vocabulary = {word: 0 for word in vocab}
    
    f_counter = 0
    for word, value in vocabulary.items():
        doc_count = 0

        for i, doc in enumerate(corpus):
            if word in doc:
                doc_count += 1
                

        vocabulary[word] = np.log((len(corpus)/float(doc_count)))
        f_counter += 1
        if verbose:
            if f_counter % 1000 == 1:
                print(f'Number of words IDFed: {f_counter}')
        
    return vocabulary

def tf_idf(stemmed_corpus: list) -> list:
    """
    (WIP)
    Caclulates the Term Frequency - Inverse Document Frequency for all documents in a corpus. TF-IDF is the relative term frequency of a word in a particularparticular document multiplied by its inverse document frequency for the corpus. 
    
    Parameters:
    stemmed_corpus (list): A list of documents containing stemmed tokens. 

    Returns:
    list: Returns a list of dictionaries, each dictionary representing a document in the corpus with the tf_idf score for each word in the document. 
    """
    output = []
    
    tfed = [tf(doc) for doc in stemmed_corpus]
    idfed = idf(stemmed_corpus)
    
    for doc in tfed:
        for word in doc:
            doc[word] *= idfed[word]
        
        
    return tfed

### Data Handling ###

def resample_majority(df: pd.DataFrame, criteria: str, majority_class: int) -> pd.DataFrame:
    """
    Reduces the size of a particular majority class in a dataframe in line with the mean of size of the remaining classes.

    Parameters:
    df (pd.DataFrame): The pandas dataframe to be downsampled.
    criteria (str): The column in the dataframe to be downsampled.
    majority_class (int): The class to be downsampled.


    Returns:
    pd.DataFrame: The resampled dataframe.
    """
    majority_df = df[df[criteria] == majority_class]
    minority_df = df[df[criteria] != majority_class]
    minority_mean = int(minority_df[criteria].value_counts().mean())
    resampled_majority = resample(majority_df, replace = False, n_samples = minority_mean, random_state = 42)
    reassembled_df = shuffle(pd.concat([resampled_majority, minority_df]), random_state=42)
    return reassembled_df.reset_index(drop=True)


### Metrics for finer evaluation of keras models ###

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

### Verbose evaluation of keras models ###

def verbose_evaluate(model, X_test, y_test):
    metrics = model.metrics_names
    metric_scores = model.evaluate(X_test, y_test)
    score_dict = {}
    for i, metric in enumerate(metric_scores):
        print(f'The {metrics[i]} score is: {metric}')
        score_dict.update({metrics[i]: metric})
    return score_dict
