import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unidecode import unidecode
import contractions
import nltk
from emot.emo_unicode import EMOTICONS_EMO
# define emoticons not provided in package
EMOTICONS_EMO[';-)'] = 'Wink'
EMOTICONS_EMO[': )'] = 'smile face'
EMOTICONS_EMO['\m/'] = 'happy'

PATH = 'data-csv/'

# ----------read data
df_raw = pd.read_csv(PATH+'randomdata.csv', header=0, index_col=[0])
key_col = ['overall', 'reviewText', 'summary', 'type', 'asin']
df = df_raw[key_col]

# ----------check null value
print('missing values exists:\n')
print(df.isnull().sum())
# remove null
df = df.dropna()


# ----------clean text
def clean_text(x, punc=True):
    # ----------convert text to ASCII form
    x = unidecode(x)
    # ----------lowercase
    # x = x.lower()
    # remove contraction
    x = contractions.fix(x)
    # convert emoticons into text
    for emot in EMOTICONS_EMO:
        x = x.replace(emot, EMOTICONS_EMO[emot])
    # # convert A/A+/A++ .etc into text
    # MARK = {r'[Aa]': 'good', r'[Aa][\+]+': 'Great!'}
    # for m in MARK:
    #     x = re.sub(m, MARK[m], x)
    # ----------remove url(not-well formatted)
    # match_url = re.compile(r'http\S+')
    match_url = re.compile(r'https?://(www\.)?([-_\w\s\.\/]*)')
    x = re.sub(match_url, "", x)
    # ----------remove consecutive letter 3ormore
    # x = re.sub(r'([^\W\d_])\1{2,}', r'\1\1', x)
    # ----------remove parenthesis
    # x = re.sub(re.compile(r'\([^\)]*\)'), "", x)
    # x = re.sub(re.compile(r'[()]'), "", x)
    # remove special chars
    x = re.sub(r'[^a-zA-z0-9.,$!?/:;\"\'\s]', "", x)
    # remove non english chars
    x = re.sub(r'[^\x00-\x7f]',"", x)
    # remove punc
    if punc:
        x = re.sub(r'[^a-zA-z0-9$\s]', "", x)
    return x

# keep punctuation
df['text'] = df['reviewText'].astype(str).apply(clean_text, punc=False)
# df['words'] = df['reviewText'].astype(str).apply(clean_text, punc=True)

# ----------text length visual
raw_txt_len = df.text.apply(lambda x: len(x.split()))
sum_len = df.summary.apply(lambda x: len(x.split()))

plt.figure()
plt.hist(raw_txt_len, bins=40, color='skyblue')
plt.title('Review Token Length')
plt.xticks(np.arange(0, 900, 100))
# plt.xlabel(np.arange(0, 900, 100))
plt.xlim((0, 900))
plt.xlabel('length')
plt.ylabel('count')
plt.grid(axis='x')
plt.show()

plt.figure()
plt.hist(sum_len, bins=20, color='skyblue')
plt.title('Summary Token Length')
plt.xticks(np.arange(0, 100, 10))
plt.xlabel('length')
plt.ylabel('count')
plt.grid(axis='x')
plt.show()

# ----------remove too long text
def rm_long(x, max_len=1000):
    """
    truncate data longer than max_len
    :param x: str input text
    :param max_len: int
    :return:
    """
    words = x.split()
    if len(words) > max_len:
        truncate = words[:max_len]
        return " ".join(truncate)
    else:
        return x

df['text'] = df['text'].astype(str).apply(rm_long, max_len=1000)
# df['words'] = df['words'].astype(str).apply(rm_long)
# ----------remove non-sense text
print('Dataset has text with no sense:')
print(df[df.text==""])
df = df[df.text!=""].reset_index()

# ----------remove stop words
# nltk.download('stopwords')
# stop_words = nltk.corpus.stopwords.words('english')
#
# def remove_stop_words(corpus):
#     result = []
#     corp = corpus.split(' ')
#     result = [w for w in corp if w not in stop_words]
#     result = " ".join(result).strip()
#
#     return result

# df['words'] = df['words'].apply(remove_stop_words)

# ----------overview
print(f'shape of cleaned data:{df.shape}')

# ----------train test split
SEED=666
TEST_RATIO = 0.2
# N = 50000
N = len(df)
OUTPUT_PATH = 'clean_data/'
df_sub = df.sample(N, random_state=SEED)
df_asin = df[['text','summary','asin']].groupby('asin').agg({'text': lambda x: " ".join(x), 'summary': lambda x: ". ".join(x)})
train_set = df_sub.iloc[:int(len(df_sub) * (1 - TEST_RATIO))][['text', 'summary']]
test_set = df_sub.iloc[-int(len(df_sub) * TEST_RATIO):][['text', 'summary', 'asin']]
print(f'shape of train data:{train_set.shape}')
print(f'shape of test data:{test_set.shape}')
train_set.to_csv(OUTPUT_PATH+'train.csv')
test_set.to_csv(OUTPUT_PATH+'test.csv')
df_asin.to_csv(OUTPUT_PATH+'item_reviews.csv')
