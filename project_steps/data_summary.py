import pandas as pd 



def data_summary(train_images_flat, train_labels):
    train_df = pd.DataFrame(train_images_flat)
    train_df['label'] = train_labels
    train_summary = train_df.describe()
    # print(train_summary)
    with open('data_summary.md', 'w') as file:
        file.write(train_summary.to_markdown())
    print(train_summary)
    