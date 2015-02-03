import pandas as pd

# Removes weird spacing in 'sentiment' and
# 'target' columns
def cleanData(input_name, output_name):
    df = pd.read_csv(input_name)

    # Solves: TypeError: descriptor 'strip' requires a 'str'
    # object but received a 'float'
    df.fillna("",inplace=True) 

    # Clean up data (some categories have weird spacing)
    df['sentiment'] = df['sentiment'].map(str.strip)
    df['target'] = df['target'].map(str.strip)

    # output edited csv
    df.to_csv(output_name, na_rep="0", index=False)

def main():
    # Train
    train_in_name = "train.csv"
    train_out_name = "fix_train.csv"

    # Dev
    dev_in_name = "dev.csv"
    dev_out_name = "fix_dev.csv"

    # Test
    test_in_name = "test.csv"
    test_out_name = "fix_test.csv"

    cleanData(train_in_name, train_out_name)
    cleanData(dev_in_name, dev_out_name)
    cleanData(test_in_name, test_out_name)

main()