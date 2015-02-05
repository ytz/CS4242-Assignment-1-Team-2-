import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re


def preprocess(df, copy=False):
    df.fillna("",inplace=True)
    porter_stemmer = PorterStemmer()

    # Iterate tweets
    for index, row in df.iterrows():
        # Retrieve tweet for a particular row
        tweet = row['content']

        

        # Stemming (not sure if necessary or not)
        porter_stemmer.stem(tweet)

        # TO-DO: abbrevation/emoticons replaced by actual meaning

        # TO-DO: Hashtag

        # TO-DO: RT/@/URL
        tweet = re.sub("RT", "", tweet)
        # http://stackoverflow.com/questions/2304632/regex-for-twitter-username
        tweet = re.sub('(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z_]+[A-Za-z0-9_]+)', "", tweet)
        tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "url", tweet)
        tweet = re.sub("['\":,]", "", tweet)
        
        # lower-caps
        tweet = tweet.lower()

        # POS tag

        # Tokenise
        df = tokenise(df, tweet, index, copy)



    # remove 'content' column
    df = df.drop('content',1)

    return df

def tokenise(df, tweet, index, copy):
    # Tokenise string (tweet)
    words = nltk.word_tokenize(tweet)  
    # remove stopwords
    words = [w for w in words if not w in stopwords.words('english')]

    # TO-DO: Negation?

    # Unigram (create column if word exist)
    if (copy == False):
        for each_word in words:
            try:
                df.loc[index, each_word] = 1 # replace cell value with 1 (presence)
            except KeyError:
                df[each_word] = 0            # create column first
                df.loc[index, each_word] = 1 # replace cell value with 1 (presence)


        # TO-DO: Bigram

    # Copy == True (for DEV and TEST)
    else:
        for each_word in words:
            if each_word in df.columns:
                df.loc[index, each_word] = 1

    return df

def copy_features(df_train, df_curr):
    column_list = list(df_train.columns.values)
    for x in range(6, len(df_train.columns)):
        df_curr[column_list[x]] = 0
    df_curr = preprocess(df_curr, copy=True)

    return df_curr


def main():
    df_train = pd.read_csv('fix_train.csv')
    df_train = preprocess(df_train)
    df_train.to_csv("preprocess_train.csv", na_rep="0",index=False)

    # Tokenise 'Dev' and 'Test' using exact features from 'Train'
    df_dev = pd.read_csv('fix_dev.csv')
    df_dev = copy_features(df_train, df_dev)
    df_dev.to_csv("preprocess_dev.csv", na_rep="0",index=False)

    df_test = pd.read_csv('fix_test.csv')
    df_test = copy_features(df_train, df_test)
    df_test.to_csv("preprocess_test.csv", na_rep="0",index=False)


main()