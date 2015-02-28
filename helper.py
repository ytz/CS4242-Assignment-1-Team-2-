from sklearn import feature_extraction
import pandas as pd
import pickle

def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
        Modified from https://gist.github.com/kljensen/5452382
    """
    vec = feature_extraction.DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].to_dict(outtype='records')).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData)

def format_dataframe(dataframe):
	""" Take a dataframe and do several things:
		1. Remove irrelevant columns
		2. Change 'sentiment' column to 'category' type instead
		   of 'string'
	    3. One-hot encoding 'target' column

	    Returns:
	    target (what we want to predict) and, 
	    features (list of features)
	"""
	dataframe.fillna(0,inplace=True)
	# Remove irrelevant columns
	del dataframe["user id"]
	del dataframe["tweet id"]
	del dataframe["username"]
	del dataframe["annotator id"]
	del dataframe["target"]

	# change 'sentiment' to category type instead of string
	#dataframe["sentiment"] = dataframe["sentiment"].astype('category') 
	# One-hot encoding 'target'
	#dataframe, _= one_hot_dataframe(dataframe, ['target'], replace=True)
	# One-hot encoding 'user_location'
	dataframe, _= one_hot_dataframe(dataframe, ['user_location'], replace=True)
	del dataframe["user_location"]

	target = dataframe["sentiment"]
	del dataframe["sentiment"]
	features = dataframe.as_matrix()

	return target, features

def file_to_array_pickle(file_location, pickle_name):
	""" Read a text file and convert into array.
		Convert this array into a pickle file.
	""" 
	f = open(file_location, 'r')
	array = []
	for line in f:
		array.append(line.strip())

	# Save array into a pickle file
	pickle.dump(array, open(pickle_name+".p", "wb"))

	f.close()

def open_array_pickle(pickle_location):
	return pickle.load(open(pickle_location,"rb"))

def getPositiveFreq(words):
	list = ['tell','publicoption','support','pass','call','topprog']
	score = 0
	for each_word in words:
	    if (each_word in list):
	        score += 1
   	return score

def getNegativeFreq(words):
	list = ['gop','sgp','teaparty','tlot','obamacare','don\'t', 'ocra']
	score = 0
	for each_word in words:
	    if (each_word in list):
	        score += 1
   	return score

def getNeutralFreq(words):
	list = ['house','rep','$','dc']
	score = 0
	for each_word in words:
	    if (each_word in list):
	        score += 1
   	return score

def getConservativeBoolean(words):
	list = ['conservative','conservatarian','republican',
			'patriot','american','teaparty','tea']
	for each_word in words:
	    if (each_word in list):
	        return 1
	return 0
