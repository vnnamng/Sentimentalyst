import pandas as pd
import emoji
import string
from langdetect import detect



pd.set_option('display.max_columns', None)
# Specify the file path
file_path = 'dataset/steam_dataset_trunc.csv'

# Read the CSV file
data = pd.read_csv(file_path)

def clean_data(df):
  pd.options.mode.copy_on_write = True
  
  # Remove rows with empty review_text
  df = df[df['review_text'].notnull()]

  # Remove emoji rows
  df['review_text'] = df['review_text'].apply(lambda x: emoji.replace_emoji(x,''))
  
  # Remove punctuation
  df['review_text'] = df['review_text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
  
  # Replace all row that has non ASCII with empty string
  df['review_text'] = df['review_text'].apply(lambda x: x if x == ''.join([i if ord(i) < 128 else '' for i in x]) else "")

  # Remove rows with empty review_text
  df = df[df['review_text'] != '']
  # Remove rows that are not in English
  for index, row in df.iterrows():
    try:
      if detect(row['review_text']) != 'en':
        df.drop(index, inplace=True)
    except:
      df.drop(index, inplace=True)

  return df

clean_data = clean_data(data)
print(clean_data.describe())
print(clean_data.head())
