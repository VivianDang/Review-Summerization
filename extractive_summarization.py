import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize
from collections import Counter
nltk.download('punkt')

PATH = 'data-csv/'
df = pd.read_csv(PATH+'clean_data.csv', header=0, index_col=[0])

def get_bag_of_words(text, counter):
    list_of_words = word_tokenize(text)
    # for word in list_of_words:
    #     if word in counter:
    #         counter[word] += 1
    #     else:
    #         counter[word] = 1
    counter.update(list_of_words)

def get_sent_value(text, sentence_value):
    list_sent = sent_tokenize(text)
    for s in list_sent:
        for word, freq in bog.items():
            if word in s:
                if s in sentence_value:
                    sentence_value[s] += freq
                else:
                    sentence_value[s] = freq

bog = Counter()
sentence_value = {}
df.text.apply(get_bag_of_words, counter=bog)
df.text[:100].apply(get_sent_value, sentence_value=sentence_value)

#-----------------TextRank
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Creating text parser using tokenization
text = df.text[4578]
def text_rank_method(text):

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Summarize using sumy TextRank
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 2)
    text_summary = ""
    for sentence in summary:
      text_summary += str(sentence)
    return text_summary

df['txtrank_sum'] = df.text[:100].apply(text_rank_method)

#-----------------Luhn
from sumy.summarizers.luhn import LuhnSummarizer
def luhn_method(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer_luhn = LuhnSummarizer()
    summary_1 = summarizer_luhn(parser.document, 2)
    dp = []
    for i in summary_1:
        lp = str(i)
        dp.append(lp)
    final_sentence = ' '.join(dp)
    return final_sentence

df['luhn_sum'] = df.text[:100].apply(luhn_method)

#-----------------LSA
from sumy.summarizers.lsa import LsaSummarizer
def lsa_method(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer_lsa = LsaSummarizer()
    summary_2 = summarizer_lsa(parser.document, 2)
    dp = []
    for i in summary_2:
        lp = str(i)
        dp.append(lp)
    final_sentence = ' '.join(dp)
    return final_sentence

df['lsa_sum'] = df.text[:100].apply(lsa_method)