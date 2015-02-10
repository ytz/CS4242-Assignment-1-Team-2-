import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re
from nltk.util import ngrams


def preprocess(df, copy=False):
    df.fillna("",inplace=True)
    porter_stemmer = PorterStemmer()

    # Iterate tweets
    for index, row in df.iterrows():
        # Retrieve tweet for a particular row
        tweet = row['content']

        # Stats Collection
        df = collectStats(df, tweet, index)

        # lower-caps
        tweet = tweet.lower()

        
        
        

        # Stemming (not sure if necessary or not)
        #porter_stemmer.stem(tweet)

        # TO-DO: abbrevation/emoticons replaced by actual meaning

        # TO-DO: Hashtag

        # TO-DO: RT/@/URL
        tweet = re.sub("rt", "", tweet)
        # http://stackoverflow.com/questions/2304632/regex-for-twitter-username
        tweet = re.sub('(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z_]+[A-Za-z0-9_]+)', "", tweet)
        tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "url", tweet)
        tweet = re.sub("['\":,]", "", tweet)
        
        

        # POS tag

        # Tokenise
        df = tokenise(df, tweet, index, copy)



    # remove 'content' column
    df = df.drop('content',1)

    return df

def tokenise(df, tweet, index, copy):
    # Tokenise string (tweet)
    words = nltk.word_tokenize(tweet)  
    # Sentiment Analysis
    df = sentimentLexicon(df, words, index)    
    # remove stopwords
    words = [w for w in words if not w in stopwords.words('english')]

    # tweet without stopwords
    tweet_nostop = ' '.join(words)

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
        bigrams = ngrams(tweet_nostop.split(), 2)
        for grams in bigrams:
            try:
                df.loc[index, grams[0] + ' ' + grams[1]] = 1 # replace cell value with 1 (presence)
            except KeyError:
                df[grams] = 0            # create column first
                df.loc[index, grams[0] + ' ' + grams[1]] = 1 # replace cell value with 1 (presence)

    # Copy == True (for DEV and TEST)
    else:
        for each_word in words:
            if each_word in df.columns:
                df.loc[index, each_word] = 1

        bigrams = ngrams(tweet_nostop.split(), 2)
        for grams in bigrams:
            if grams in df.columns:
                df.loc[index, grams[0] + ' ' + grams[1]] = 1

    return df

def sentimentLexicon(df, words, index):
    # Create Pickle Files
    file_to_array_pickle("positive.txt", "positive")
    file_to_array_pickle("negative.txt", "negative")

    positive_list = open_array_pickle("positive.p")
    negative_list = open_array_pickle("negative.p")

    positive_score = 0
    negative_score = 0

    for each_word in words:
        if (each_word in positive_list)
            positive_score += 1
        if (each_word in negative_list)
            negative_score += 1

    # Enter Positive Score to Dataframe
    try:
        df.loc[index, "no_of_positve_word"] = positive_score
    except KeyError:
        df["no_of_positve_word"] = 0            # create column first
        df.loc[index, "no_of_positve_word"] = positive_score # replace cell value

    # Enter Negative Score to Dataframe
    try:
        df.loc[index, "no_of_negative_word"] = negative_score
    except KeyError:
        df["no_of_negative_word"] = 0            # create column first
        df.loc[index, "no_of_negative_word"] = negative_score # replace cell value 

    return df 

def collectStats(df, tweet, index):
    # Number of capital letters
    no_of_capital = sum(1 for c in tweet if c.isupper())

    try:
        df.loc[index, "no_of_capital_letters"] = no_of_capital
    except KeyError:
        df["no_of_capital_letters"] = 0            # create column first
        df.loc[index, "no_of_capital_letters"] = no_of_capital # replace cell value

    # Number of '!' & '?'
    no_of_exclam = tweet.count('!') + tweet.count('?')

    try:
        df.loc[index, "no_of_exclaim"] = no_of_exclam
    except KeyError:
        df["no_of_exclaim"] = 0            # create column first
        df.loc[index, "no_of_exclaim"] = no_of_exclam # replace cell value

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