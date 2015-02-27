import tweepy
"""
consumer_key = "rPvXh8Hx0fm9u7gYopN0AAaCb"
consumer_secret = "yGeTiJMP6lSDY6BMc1pzXTyusigYwyhkyG3zlNYsDGXmPOQUYv"
access_token = "40458142-iKgdSgtsA8YmvNHv4xGRsMx81jTkz7YnPqhzSE5rs"
access_token_secret = "9UNSp6AcRXVmQz8vnCuMOlZvW9PUPPHVCAQDGbGS78Ndh"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

status = api.get_status("10237553563")

print status.user.description
"""

def getAPI():
	consumer_key = "rPvXh8Hx0fm9u7gYopN0AAaCb"
	consumer_secret = "yGeTiJMP6lSDY6BMc1pzXTyusigYwyhkyG3zlNYsDGXmPOQUYv"
	access_token = "40458142-iKgdSgtsA8YmvNHv4xGRsMx81jTkz7YnPqhzSE5rs"
	access_token_secret = "9UNSp6AcRXVmQz8vnCuMOlZvW9PUPPHVCAQDGbGS78Ndh"

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)

	return api

def inputUserBio(df, api, user_id, index):
	try:
		user = api.get_user(user_id)
		user_bio = user.description

	except tweepy.error.TweepError:
		user_bio = ""

	try:
	    df.loc[index, "user_bio"] = user_bio
	except KeyError:
	    df["user_bio"] = 0            # create column first
	    df.loc[index, "user_bio"] = user_bio # replace cell value 
    
	return df
