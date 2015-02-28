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

def getSuzzieAPI():
	consumer_key = "2Jyn96zhsliDpjiOcqZnCNwqX"
	consumer_secret = "KoDLlSjIl2e8S7xBkMkbdbk5KjAJWC1wkp3O6Pb71JvakCaAce"
	access_token = "874954729-RhhlakcTpmTE2OpcCX6FWsLLEcxBwQZFv4MuBfig"
	access_token_secret = "KCvINjOwKS9iwj6BUhjcF5C8tF3iFTvXc5DME8J8gEqyg"

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)

	return api

def getYsAPI():
	consumer_key = "CMkbsLAV6MJnLOqZzocyzPVcQ"
	consumer_secret = "6MXATxS3ItYLVVAFRnpGBuyipVLPt1opfhOViE1EJQR2UnfvXE"
	access_token = "158665010-3EQVH9m7v1BLsSlu3SiO8leOYzViESV38FGZGdoU"
	access_token_secret = "thKZLcXa9MFpvOf5sGxy6WprhYPa8H73Pwv0suCXyOeYf"

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)

	return api

def inputUserLocation(df, api, user_id, index):
	try:
		user = api.get_user(user_id)
		user_location = user.location

	except tweepy.error.TweepError:
		user_location = ""

	try:
	    df.loc[index, "user_location"] = user_location
	except KeyError:
	    df["user_location"] = 0            # create column first
	    df.loc[index, "user_location"] = user_location # replace cell value 
    
	return df

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

def inputUserVerified(df, api, user_id, index):
	try:
		user = api.get_user(user_id)
		user_verified = user.verified

	except tweepy.error.TweepError:
		user_verified = ""

	if (user_verified == 'TRUE'):
		user_verified = 1
	elif(user_verified == 'FALSE'):
		user_verified = 0

	try:
		df.loc[index, "user_verified"] = user_verified

	except KeyError:
	    df["user_verified"] = 0            # create column first
	    df.loc[index, "user_verified"] = user_verified # replace cell value 
    
	return df


def inputUserFollowers(df, api, user_id, index):
	try:
		user = api.get_user(user_id)
		user_followers = user.follower_count

	except tweepy.error.TweepError:
		user_followers = ""

	try:
		df.loc[index, "user_followers"] = user_followers

	except KeyError:
	    df["user_followers"] = 0            # create column first
	    df.loc[index, "user_followers"] = user_followers # replace cell value 
    
	return df

def inputUserFollowing(df, api, user_id, index):
	try:
		user = api.get_user(user_id)
		user_following = user.following

	except tweepy.error.TweepError:
		user_following = ""

	try:
		df.loc[index, "user_following"] = user_following

	except KeyError:
	    df["user_following"] = 0            # create column first
	    df.loc[index, "user_following"] = user_following # replace cell value 
    
	return df