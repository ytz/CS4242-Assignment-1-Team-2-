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

def inputUserVerified(df, api, user_id, index):
	try:
		user = api.get_user(user_id)
		user_verified = user.verified

	except tweepy.error.TweepError:
		user_verified = ""

	if (user_verified == 'True'):
		user_verified = 1
	elif(user_verified == 'False'):
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