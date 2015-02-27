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

def inputUserLocation(df, api, user_id, index):
	try:
		user = api.get_user(user_id)
		user_loc = user.location

	except tweepy.error.TweepError:
		user_loc = ""

	try:
	    df.loc[index, "user_loc"] = user_loc
	except KeyError:
	    df["user_loc"] = 0            # create column first
	    df.loc[index, "user_loc"] = user_loc # replace cell value 
    
	return df

def getCoordinates(df, api, tweet_id, index):
	try:
		status = api.get_status(tweet_id)
		coordinates = status.coordinates
		place = status.place

	except tweepy.error.TweepError:
		coordinates = "None"
		place = "None"

	try:
	    df.loc[index, "coordinates"] = coordinates
	except KeyError:
	    df["coordinates"] = 0            # create column first
	    df.loc[index, "coordinates"] = coordinates # replace cell value 

	try:
	    df.loc[index, "place"] = place
	except KeyError:
	    df["place"] = 0            # create column first
	    df.loc[index, "place"] = place # replace cell value 

	return df

