import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

df = pd.read_csv('test.csv')

# Clean up data (some categories have weird spacing)

# Solves: TypeError: descriptor 'strip' requires a 'str'
# object but received a 'float'
df.fillna("",inplace=True) 

df['sentiment'] = df['sentiment'].map(str.strip)
df['target'] = df['target'].map(str.strip)

porter_stemmer = PorterStemmer()

# Iterate tweets
for index, row in df.iterrows():
    tweet = row['content']
    # Word preprocessing
    porter_stemmer.stem(tweet)
    # Tokenise tweet
    words = nltk.word_tokenize(tweet)  
    words = [w for w in words if not w in stopwords.words('english')]
    # bigram
    # add back to df while +1 for frequency
    # create column if word exist
    for each_word in words:
        df.loc[index, each_word] = 1

# remove 'content' column
df = df.drop('content',1)

# TO-DO: remove unmaed column



# output edited csv
df.to_csv("edit_test.csv", na_rep="0")