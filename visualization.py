import os
import re

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from PIL import Image
from transformers import AutoTokenizer
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
stopwords.update({'five', 'stars', 'four'})
path = os.getcwd()

log_file = pd.read_csv(path+'/top2log.csv', index_col=[0])
logs = log_file.groupby('epoch').aggregate('sum')
logs = logs[['loss', 'learning_rate', 'step', 'eval_loss', 'eval_rouge1',
       'eval_rouge2', 'eval_rougeL', 'train_loss']]
fig = plt.figure()
logs['learning_rate'].plot()
plt.show()
logs[['loss', 'eval_loss']].plot()
plt.show()

train = pd.read_csv(path+"/clean_data/train.csv", index_col=[0])

def nltk_tokenization(text, remove_punc=False):
    text = text.lower()

    # remove punc
    if remove_punc:
        text = re.sub(r'[^\w\s]','',text)
    text = nltk.word_tokenize(text)

    return text

train['tokens'] = train.text.apply(nltk_tokenization, remove_punc=True)
train['labels'] = train.summary.apply(nltk_tokenization, remove_punc=True)

def get_bag_of_words(list_of_words, counter):
    list_of_words = [w for w in list_of_words if w not in stopwords]
    counter.update(list_of_words)

def get_most_n(df, n):
    ct = Counter()
    df.apply(get_bag_of_words, counter=ct)
    return ct.most_common(n)

words = dict(get_most_n(train['tokens'], 100))
sum_words = dict(get_most_n(train['labels'], 100))
output = pd.read_csv(path+'/output/Review_Sampling_Summary.csv')
mask = np.array(Image.open(path+'/image/dialog.png'))
mask_cloud = np.array(Image.open(path+'/image/cloud.png'))

# plt.figure(figsize=[20,20])
# wc1 = WordCloud(
#     width=3000,
#     height=2000,
#     mask=mask,
#     background_color='white',
#     min_font_size=10,
#     colormap='Set2',
#     random_state=12).generate_from_frequencies(words)
# plt.imshow(wc1, interpolation="bilinear")
# plt.axis('off')
# plt.savefig(path+'/image/input.png', format="png")
# plt.show()
#
# plt.figure(figsize=[20,20])
# wc2 = WordCloud(
#     width=3000,
#     height=2000,
#     mask=mask,
#     background_color='white',
#     min_font_size=10,
#     max_words=60,
#     colormap='Set2',
#     random_state=12).generate_from_frequencies(sum_words)
# plt.imshow(wc2, interpolation="bilinear")
# plt.axis('off')
# plt.savefig(path+'/image/output.png', format="png")
# plt.show()


# output view
item_sum = pd.read_csv(path+'/output/item_summary.csv',index_col=[0])
large_sum = item_sum[item_sum['reviewText'].apply(lambda x: len(x.split())>150)]

plt.figure(figsize=[20,20])
wc3 = WordCloud(
    # width=3000,
    # height=2000,
    mask=mask,
    background_color=None,
    mode='RGBA',
    min_font_size=10,
    # max_words=60,
    colormap='Set2',
    random_state=12).generate(large_sum.iloc[0]['reviewText'])
plt.imshow(wc3, interpolation="bilinear")
plt.axis('off')
# plt.savefig(path+'/image/toy_input.png', format="png")
wc3.to_file(path+'/image/toy_input.png')
plt.show()


plt.figure(figsize=[20,20])
wc4 = WordCloud(
    width=300,
    height=200,
    mask=mask_cloud,
    background_color='white',
    min_font_size=10,
    # max_words=60,
    colormap='Set2',
    random_state=12).generate(large_sum.iloc[0]['predSumary'])
plt.imshow(wc4, interpolation="bilinear")
plt.axis('off')
plt.savefig(path+'/image/toy_output.png', format="png")

plt.show()


plt.figure(figsize=[20,20])
wc5 = WordCloud(
    # width=3000,
    # height=2000,
    mask=mask,
    background_color=None,
    mode='RGBA',
    min_font_size=10,
    # max_words=60,
    colormap='Set2',
    random_state=12).generate(large_sum.iloc[8]['reviewText'])
plt.imshow(wc5, interpolation="bilinear")
plt.axis('off')
# plt.savefig(path+'/image/book_input.png', format="png")
wc5.to_file(path+'/image/book_input.png')
plt.show()


plt.figure(figsize=[20,20])
wc6 = WordCloud(
    width=300,
    height=200,
    mask=mask_cloud,
    background_color='white',
    min_font_size=10,
    # max_words=60,
    stopwords=None,
    colormap='Set2',
    random_state=12).generate(large_sum.iloc[8]['predSumary'])
plt.imshow(wc6, interpolation="bilinear")
plt.axis('off')
plt.savefig(path+'/image/book_output.png', format="png")
plt.show()