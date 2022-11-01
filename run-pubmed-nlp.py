#!/usr/bin/env python3
"""
Author : Rory Maclean <rorymcln@gmail.com>
Date   : 2022-11-01
"""

import argparse
import numpy as np
import pandas as pd
import re
import nltk             #nb need to download datasets on command line when first using nltk
import unicodedata2 as unicodedata
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)


CUSTOM_STOPWORDS = [
'conflict',
'interest',
'statement',
'artificial',
'learning',
'deep',
'intelligence',
'machine',
'model',
'clinical',
'trial',
'data',
'set',
'clinical',
'practice',
'test',
'outcome',
'95',
'ci',
'confidence',
'interval',
'controlled',
'development',
'area',
'auc',
'curve',
'receiver',
'characteristic',
'grant',
'fee',
'personal',
'medicine',
'precision']


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='''Pipeline to process pubmed text file download:
        extracting abstract, removing stopwords, lemmatization,
        ngram calculation and plotting
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file_path',
                        metavar='FILE',
                        help='A text file downloaded from pubmed',
                        type=argparse.FileType('rt'))


    parser.add_argument('--min',
                        help='minimum text block length',
                        metavar='int',
                        type=int,
                        default=1000)

    parser.add_argument('--max',
                        help='maximum text block length',
                        metavar='int',
                        type=int,
                        default=3000)

    return parser.parse_args()


# --------------------------------------------------
def by_length(strings, min, max):
    """
    Function to select string blocks over a threshold length
    as a heuristic that the abstract is the longest string
    block out of the blocks per pubmed entry.
    Need to tune min and max; max introduced for a single entry
    which had a very long block of authors as it was an
    entry for an entire conference with every presentation listed
    """
    return [string for string in strings if len(string) > min and len(string) < max]


# --------------------------------------------------
def remove_author_info(strings):    
    """
    Function to remove the author information block as this 
    is often quite long and not filtered out in the prior 
    by_length function in the pipeline
    """
    return [string for string in strings if not re.search(r'^Author information', string)]


# --------------------------------------------------
def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english') + CUSTOM_STOPWORDS
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  lemma = [wnl.lemmatize(word) for word in words if word not in stopwords]
  return [word for word in lemma if word not in stopwords]

# --------------------------------------------------
def main():

    args = get_args()

    min_arg = args.min
    max_arg = args.max
    file_arg = args.file_path

    # print(f'min_arg = "{min_arg}"')
    # print(f'max_arg = "{max_arg}"')
    # print(f'file_arg = "{file_arg.name}"')

    with open(file_arg.name, 'r') as f:
        text = f.read()
        f.close()

    text = text.split('\n\n') # text blocks are separated by two newlines

    text = remove_author_info(text)

    abstracts = by_length(text, min_arg, max_arg)

    words = basic_clean(' '.join(abstracts))

    bigrams = pd.Series(nltk.ngrams(words, 2))
    trigrams = pd.Series(nltk.ngrams(words, 3))

    bigrams_select = bigrams.value_counts()[:20]
    trigrams_select = trigrams.value_counts()[:20]

    bigrams_select.sort_values().plot.barh(color='blue', width=.9, figsize=(12,8))
    plt.title('20 Most Frequently Occuring Bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# of Occurences')

    plt.show()

    trigrams_select.sort_values().plot.barh(color='blue', width=.9, figsize=(12,8))
    plt.title('20 Most Frequently Occuring Trigrams')
    plt.ylabel('Trigram')
    plt.xlabel('# of Occurences')

    plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
