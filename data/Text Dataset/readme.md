# Dataset Overview #

- The dataset consists of Reddit posts from SuicideWatch, depression, and teenagers subreddits, collected using the Pushshift API. Posts are labeled as suicide or non-suicide, with 116,037 examples for each class, making it perfectly balanced.

## Cleaning and Preprocessing ##

The text data was carefully cleaned:

- Lowercased and normalized

- Removed URLs, special characters, and extra spaces

- Fixed typos and separated merged words

- Kept important words like "no" and "not" when removing stopwords

- Dropped meaningless words like "filler" and filtered out very long posts (over 62 words)

This cleaned dataset was used to train the DistilBERT + CNN hybrid model for detecting signs of suicidal ideation.


---
