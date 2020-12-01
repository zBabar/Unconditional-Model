import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from collections import defaultdict, namedtuple
import re

data=pd.read_csv('/media/zaheer/Data/Image_Text_Datasets/MIMIC_CXR/csv_files/mimic_cxr_sectioned.csv')

print(data.findings[0])

data=data['findings'].to_list()

print(len(data))

list_words=defaultdict()
random_lst=[]
pattern = '[0-9]'
for inst in data:
    #print(inst)
    if not pd.isnull(inst):
        inst=inst.lower()
        inst1= inst.replace(',', '').replace("'", "").replace('"', '')#.replace('.', '')
        inst1 = inst1.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').replace('___','')
        inst1=re.sub(pattern, '', inst1)
        lst=sent_tokenize(inst1)
        #lst = word_tokenize(inst1)
        #print(lst)
        for word in lst:
            if not word.isnumeric():
                random_lst.append(word)
                if not word in list_words:

                    list_words[word]=0

                list_words[word]+=1

    else:
        continue

print(set(random_lst))
print(len(list_words.keys()))
print(len(set(random_lst)))



