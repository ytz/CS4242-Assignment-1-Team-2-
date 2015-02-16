# Setup
Best if you already have [Anaconda](https://store.continuum.io/cshop/anaconda/).

Update the modules inside Anaconda using the following command:
    
    - conda update --all
    - conda update pandas
    - conda update scikit-learn
    
# Files

## data_cleaning.py
The original csv have incorrect spacing for *'sentiment'* and *'target'* columns. (e.g. ' hcr' instead of 'hcr'). This code will fix this and output the corrected csv: **fix_train.csv**, **fix_dev.csv**, **fix_test.csv**.

## preprocess.py
This code will generate features from the fixed csv. Generated feature includes:
* Positive & Negative word count
* Unigram & Bigram
* Number of capital letters
* Number of '!' & '?'

3 files will be generated: **preprocess_train.csv**. **preprocess_dev.csv**, **preprocess_test.csv**. Note that the features from *dev* and *test* are based on *train*, aka they're the same.

# train.py
This code will:
* Generate a classifier pickle file for predict.py 
* Conduct 10-fold validation on training data
* Evaluate performance on dev data

# predict.py
This code will evaluate the performance on the test data.