import os
import re
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
key_col = ['overall', 'reviewText', 'summary', 'type']
df = df_raw[key_col]

# ----------check null value
print('missing values exists:\n', df.isnull().sum())
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
plt.hist(raw_txt_len, bins=100)
plt.show()

plt.hist(sum_len,bins=10)
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
        return words[:max_len]
    else:
        return x

df['text'] = df['text'].astype(str).apply(rm_long, max_len=300)
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
df.to_csv(PATH+'clean_data.csv')

