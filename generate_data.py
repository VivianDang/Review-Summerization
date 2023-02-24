import pandas as pd
import os
import re

os.getcwd()
PATH = 'data-csv/'
os.chdir(PATH)
files = os.listdir()

N = 10000
# for d in files:
#     df = pd.read_csv(d, header=0)

usecol = ['overall', 'vote', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style', 'reviewText', 'summary']
dtypes = {'overall':'object', 'vote':'str', 'verified':'bool','reviewTime':'object', 'reviewerID':'str', 'asin':'str', 'style':'object',
          'reviewText':'str', 'summary':'str'}

df = pd.read_csv(files[0], header=0, usecols=usecol,dtype=dtypes)
samp = df.sample(n=N)
samp['type'] = re.search(r'[A-Za-z_]*', files[0]).group()[:-1]

for i in range(1, len(files)):
    print(files[i])
    # if files[i] == 'Books_5.csv':
    df1 = pd.read_csv(files[i], header=0, usecols=usecol, dtype=dtypes, encoding='utf-8', engine="python",
                    error_bad_lines=False, )
    # else:
    #     df1 = pd.read_csv(files[i], header=0, usecols=usecol, dtype=dtypes, encoding='utf-8')

    if len(df1) < N:
        samp1 = df1.sample(n=len(df1))
        samp1['type'] = re.search(r'[A-Za-z_]*', files[i]).group()[:-1]
    else:
        samp1 = df1.sample(n=N)
        samp1['type'] = re.search(r'[A-Za-z_]*', files[i]).group()[:-1]
    samp = pd.concat([samp, samp1])

samp.to_csv('randomdata.csv')