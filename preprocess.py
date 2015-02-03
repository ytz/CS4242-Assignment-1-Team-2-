import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd


def preprocess(df):
    porter_stemmer = PorterStemmer()

    # Iterate tweets
    for index, row in df.iterrows():
        # Retrieve tweet for a particular row
        tweet = row['content']

        # Stemming
        #porter_stemmer.stem(tweet)

        # abbrevation replaced by actual meaning


        # Tokenise
        df = tokenise(df, tweet)

    # remove 'content' column
    df = df.drop('content',1)

def tokenise(df, tweet):
    # Tokenise string (tweet)
    words = nltk.word_tokenize(tweet)  
    # remove stopwords
    words = [w for w in words if not w in stopwords.words('english')]

    # Unigram (create column if word exist)
    for each_word in words:
        df.loc[index, each_word] = 1

    return df


def main():
    df_train = pd.read_csv('fix_train.csv')
    df_train = preprocess(df_train)
    df_train.to_csv("preprocess_train.csv", index=False, na_rep="0")



main()