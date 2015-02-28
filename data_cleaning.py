import pandas as pd
import twitter_data

# Removes weird spacing in 'sentiment' and
# 'target' columns
def cleanData(input_name, output_name,api):
    df = pd.read_csv(input_name)

    # Solves: TypeError: descriptor 'strip' requires a 'str'
    # object but received a 'float'
    df.fillna("",inplace=True) 

    # Clean up data (some categories have weird spacing)
    df['sentiment'] = df['sentiment'].map(str.strip)
    df['target'] = df['target'].map(str.strip)

    # Remove sentiment that are not 'positive','negative','neutral'
    df = df[(df.sentiment == 'positive') | 
            (df.sentiment == 'negative') |
            (df.sentiment == 'neutral' )]

    # Retrieve User location
    for index, row in df.iterrows():
        df = twitter_data.inputUserLocation(df, api, row['user id'], index)
    
    
    # Retrieve User Bio
    for index, row in df.iterrows():
        df = twitter_data.inputUserBio(df, api, row['user id'], index)
    

    # Retrieve User Verified
    for index, row in df.iterrows():
        df = twitter_data.inputUserVerified(df, api, row['user id'], index)


    # output edited csv
    df.to_csv(output_name, na_rep="0", index=False, encoding='utf-8')

def main():
    api = twitter_data.getAPI()
    # Train
    train_in_name = "train.csv"
    train_out_name = "fix_train.csv"

    # Dev
    dev_in_name = "dev.csv"
    dev_out_name = "fix_dev.csv"

    # Test
    test_in_name = "test.csv"
    test_out_name = "fix_test.csv"

    print 'train'
    cleanData(train_in_name, train_out_name,api)
    print 'dev'
    cleanData(dev_in_name, dev_out_name,api)
    print 'test'
    cleanData(test_in_name, test_out_name,api)

main()