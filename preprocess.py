from __future__ import division
import helper
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tag import pos_tag 
import pandas as pd
import re
from nltk.util import ngrams
import itertools
import twitter_data


def preprocess(df, copy=False):
    df.fillna("",inplace=True)
    porter_stemmer = PorterStemmer()
    #api = twitter_data.getAPI()

    # Iterate tweets
    for index, row in df.iterrows():
        print str(index+1) + '/' + str(len(df.index))

        # Retrieve user bio
        #df = twitter_data.inputUserBio(df, api, row['user id'], index)

        # Retrieve tweet for a particular row
        tweet = row['content']

        # Enter Conservative Boolean to Dataframe
        user_bio = row['user_bio']
        df = userBio(df, user_bio,porter_stemmer,index)

        # Stats Collection
        df = collectStats(df, tweet, index)

        # lower-caps
        tweet = tweet.lower()


        

        # TO-DO: abbrevation/emoticons replaced by actual meaning

        # Hashtag
        tweet = re.sub("[#]", "HASHTAG", tweet)

        # RT/@/URL
        tweet = re.sub("rt", "", tweet)
        # http://stackoverflow.com/questions/2304632/regex-for-twitter-username
        tweet = re.sub('(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z_]+[A-Za-z0-9_]+)', "", tweet)
        tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "url", tweet)
        tweet = re.sub("['\":,.!?;&-=|@()/]", "", tweet)

        # Stemming (not sure if necessary or not)
        #tweet = porter_stemmer.stem(tweet)
        
        # POS tag
        
        # Tokenise
        df = tokenise(df, tweet, index, copy)



    # remove 'content' column
    df = df.drop('content',1)
    # remove 'user_bio' column
    df = df.drop('user_bio',1)

    return df

def userBio(df, user_bio,porter_stemmer,index):
    user_bio = str(user_bio)
   
    # lower-caps
    user_bio = user_bio.lower()
    user_bio = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "url", user_bio)
    user_bio = re.sub('(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z_]+[A-Za-z0-9_]+)', "", user_bio)
    user_bio = re.sub("['\":,.!?;&-=|@()/#]", "", user_bio)
    user_bio = re.sub(r'[^\x00-\x7F]+',' ', user_bio)

    user_bio = porter_stemmer.stem(user_bio)

    words = nltk.word_tokenize(user_bio) 
    words = [w for w in words if not w in stopwords.words('english')]

    # Enter Conservative Boolean to Dataframe
    con_boolean = helper.getConservativeBoolean(words)
    try:
        df.loc[index, "con_boolean"] = con_boolean
    except KeyError:
        df["con_boolean"] = 0            # create column first
        df.loc[index, "con_boolean"] = con_boolean # replace cell value 

    return df
    

def tokenise(df, tweet, index, copy):
    # Tokenise string (tweet)
    nWords = nltk.word_tokenize(tweet)  
    # Sentiment Analysis
    df = sentimentLexicon(df, nWords, index)    
    # remove stopwords
    #words = [w for w in words if not w in stopwords.words('english')]
   
    lmtzr = WordNetLemmatizer()
    for idx,each_word in enumerate(nWords):
        # check if it is an invalid word
        if not wordnet.synsets(word_to_test):
            # remove duplicate char from word (eg. SSYYNNOOPPSSIISS)
            ''.join(ch for ch, _ in itertools.groupby(nWords[idx]))
        #lemmatize
        nWords[idx] = lmtzr.lemmatize(each_word)

    # tweet without stopwords
    tweet_nostop = ' '.join(nWords)
    words = []
    
    taggedWords = nltk.pos_tag(nWords)      #POS Tagging
    for each_tag in taggedWords:
        words.append(each_tag[0] + "/" + each_tag[1])

    # TO-DO: Negation?

    # Unigram (create column if word exist)
    if (copy == False):
        for each_word in words:
            try:
                if (isinstance( df.loc[index, each_word], int )):
                    df.loc[index, each_word] = df.loc[index, each_word] + 1 # replace cell value with 1 (presence), CHECK: Add by 1 works better?
                else:
                    df.loc[index, each_word] = 1
            except KeyError:
                df[each_word] = 0            # create column first
                df.loc[index, each_word] = 1 # replace cell value with 1 (presence)


        # TO-DO: Bigram
        bigrams = ngrams(tweet_nostop.split(), 2)
        for grams in bigrams:
            try:
                if (isinstance( df.loc[index, grams[0] + ' ' + grams[1]] , int )):
                    df.loc[index, grams[0] + ' ' + grams[1]] = df.loc[index, grams[0] + ' ' + grams[1]] + 1 # replace cell value with 1 (presence), CHECK: Add by 1 works better?
                else:
                    df.loc[index, grams[0] + ' ' + grams[1]] = 1
            except KeyError:
                df[grams] = 0            # create column first
                df.loc[index, grams[0] + ' ' + grams[1]] = 1 # replace cell value with 1 (presence)

    # Copy == True (for DEV and TEST)
    else:
        for each_word in words:
            if each_word in df.columns:
                if (isinstance( df.loc[index, each_word], int )):
                    df.loc[index, each_word] = df.loc[index, each_word] + 1 # replace cell value with 1 (presence), CHECK: Add by 1 works better?
                else:
                    df.loc[index, each_word] = 1

        bigrams = ngrams(tweet_nostop.split(), 2)
        for grams in bigrams:
            if grams in df.columns:
                if (isinstance( df.loc[index, grams[0] + ' ' + grams[1]] , int )):
                    df.loc[index, grams[0] + ' ' + grams[1]] = df.loc[index, grams[0] + ' ' + grams[1]] + 1 # replace cell value with 1 (presence), CHECK: Add by 1 works better?
                else:
                    df.loc[index, grams[0] + ' ' + grams[1]] = 1

    return df

def sentimentLexicon(df, words, index):
    # Create Pickle Files
    
    helper.file_to_array_pickle("positive.txt", "positive")
    helper.file_to_array_pickle("negative.txt", "negative")
    

    positive_list = helper.open_array_pickle("positive.p")
    negative_list = helper.open_array_pickle("negative.p")

    positive_score = 0
    negative_score = 0

    for each_word in words:
        if (each_word in positive_list):
            positive_score += 1
        if (each_word in negative_list):
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

    
    # Enter Positive Freq to Dataframe
    positive_freq = helper.getPositiveFreq(words)
    try:
        df.loc[index, "positive_freq"] = positive_freq
    except KeyError:
        df["positive_freq"] = 0            # create column first
        df.loc[index, "positive_freq"] = positive_freq # replace cell value 
    
    # Enter Negative Freq to Dataframe
    negative_freq = helper.getNegativeFreq(words)
    try:
        df.loc[index, "negative_freq"] = negative_freq
    except KeyError:
        df["negative_freq"] = 0            # create column first
        df.loc[index, "negative_freq"] = negative_freq # replace cell value 
    
    # Enter Neutral Freq to Dataframe
    neutral_freq = helper.getNeutralFreq(words)
    try:
        df.loc[index, "neutral_freq"] = neutral_freq
    except KeyError:
        df["neutral_freq"] = 0            # create column first
        df.loc[index, "neutral_freq"] = neutral_freq # replace cell value 
    
    

    return df 

def collectStats(df, tweet, index):
    # Number of capital letters
    no_of_capital = sum(1 for c in tweet if c.isupper())
    # Normalise
    no_of_capital = no_of_capital/25

    try:
        df.loc[index, "no_of_capital_letters"] = no_of_capital
    except KeyError:
        df["no_of_capital_letters"] = 0            # create column first
        df.loc[index, "no_of_capital_letters"] = no_of_capital # replace cell value

    # Number of '!' & '?'
    no_of_exclam = tweet.count('!') + tweet.count('?')
    # Normalise
    no_of_exclam = no_of_exclam/2

    try:
        df.loc[index, "no_of_exclaim"] = no_of_exclam
    except KeyError:
        df["no_of_exclaim"] = 0            # create column first
        df.loc[index, "no_of_exclaim"] = no_of_exclam # replace cell value

    """
    # Number of '#', place more emphasis with tweets with more hashtags
    no_of_hashtag = tweet.count('#')
    # Normalise
    no_of_hashtag = no_of_hashtag/2

    try:
        df.loc[index, "no_of_hashtag"] = no_of_hashtag
    except KeyError:
        df["no_of_exclaim"] = 0            # create column first
        df.loc[index, "no_of_hashtag"] = no_of_hashtag # replace cell value
    """
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
    df_train.to_csv("preprocess_train.csv", na_rep="0",index=False,encoding='utf-8')

    # Tokenise 'Dev' and 'Test' using exact features from 'Train'
    df_dev = pd.read_csv('fix_dev.csv')
    df_dev = copy_features(df_train, df_dev)
    df_dev.to_csv("preprocess_dev.csv", na_rep="0",index=False,encoding='utf-8')

    df_test = pd.read_csv('fix_test.csv')
    df_test = copy_features(df_train, df_test)
    df_test.to_csv("preprocess_test.csv", na_rep="0",index=False,encoding='utf-8')


main()