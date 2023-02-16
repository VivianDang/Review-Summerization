import json

import pandas as pd
import gzip
import os

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')
  #   return pd.read_json(d)
#
# # df = getDF('reviews_Video_Games.json.gz')
#
PATH = 'data/'
os.chdir(PATH)
files = os.listdir(PATH)

# # data1 = parse(files[0])

# with gzip.open(files[0], 'r') as fin:
#   data = json.loads(fin.read().decode('utf-8'))

