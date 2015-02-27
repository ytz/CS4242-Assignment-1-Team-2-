import helper
import nltk
import pandas as pd
import re
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import twitter_data

#helper.file_to_array_pickle("positive.txt", "positive")
#helper.file_to_array_pickle("negative.txt", "negative")

def userBio():
	df = pd.read_csv("preprocess_train.csv")

	df_positive = df[df.sentiment == 'positive']
	df_negative = df[df.sentiment == 'negative']
	df_neutral = df[df.sentiment == 'neutral']

	positive_tweets = ""
	negative_tweets = ""
	netural_tweets = ""

	positive_tweets = joinString(df_positive, positive_tweets)
	negative_tweets = joinString(df_negative, negative_tweets)
	netural_tweets = joinString(df_neutral, netural_tweets)

	words = nltk.tokenize.word_tokenize(positive_tweets)
	fdist = FreqDist(words)

	print fdist.items()[:20]
	print "\n"

	words = nltk.tokenize.word_tokenize(negative_tweets)
	fdist = FreqDist(words)

	print fdist.items()[:20]
	print "\n"

	words = nltk.tokenize.word_tokenize(netural_tweets)
	fdist = FreqDist(words)

	print fdist.items()[:20]


def wordFreq():
	df = pd.read_csv("fix_train.csv")
	df.fillna("",inplace=True)

	df_positive = df[df.sentiment == 'positive']
	df_negative = df[df.sentiment == 'negative']
	df_neutral = df[df.sentiment == 'neutral']

	positive_tweets = ""
	negative_tweets = ""
	netural_tweets = ""

	positive_tweets = joinString(df_positive, positive_tweets)
	negative_tweets = joinString(df_negative, negative_tweets)
	netural_tweets = joinString(df_neutral, netural_tweets)

	words = nltk.tokenize.word_tokenize(positive_tweets)
	fdist = FreqDist(words)

	print fdist.items()[:20]
	print "\n"

	words = nltk.tokenize.word_tokenize(negative_tweets)
	fdist = FreqDist(words)

	print fdist.items()[:20]
	print "\n"

	words = nltk.tokenize.word_tokenize(netural_tweets)
	fdist = FreqDist(words)

	print fdist.items()[:20]


def joinString(df, tweets):
	porter_stemmer = PorterStemmer()
	for index, row in df.iterrows():
		#tweet = row['content']
		tweet = row['user_bio']
		tweet = str(tweet)
		tweet = porter_stemmer.stem(tweet)

		# lower-caps
		tweet = tweet.lower()
		tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "url", tweet)
		tweet = re.sub('(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z_]+[A-Za-z0-9_]+)', "", tweet)
		tweet = re.sub("['\":,.!?;&-=|@()/#]", "", tweet)

		words = nltk.word_tokenize(tweet) 
		words = [w for w in words if not w in stopwords.words('english')]
		tweet = ' '.join(words)

		tweets = tweets + " " + tweet

	return tweets

#main()
#userBio()
myFile = "fix_dev.csv"
df = pd.read_csv(myFile)
api = twitter_data.getAPI()
for index, row in df.iterrows():
	if (row['user_bio'] == "0"):
	    df = twitter_data.inputUserBio(df, api, row['user id'], index)
df.to_csv(myFile, na_rep="0",index=False,encoding='utf-8')