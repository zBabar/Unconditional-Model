from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk


def docTo_Mat(texts,ngram):
   #texts=[clean_data(text) for text in texts]
    vectorizer = CountVectorizer(ngram_range=ngram,min_df=1,lowercase=True)
    X = vectorizer.fit_transform(texts)

    # print(X.shape, len(vectorizer.vocabulary_))


    return X,vectorizer.vocabulary_
ngrams=[1,2,3,4]
for i in ngrams:

    X,vocabulary=docTo_Mat(['mediastinal and a pulmonary edema  cardiomediastinal  silhouette  is  unchanged  with  the lungs are clear there is seen in the cardiac and hilar contours are no pleural effusion or pneumothorax is normal there is no focal consolidation to the right lung atelectasis of the left heart size of the chest'],(i,i))

    print(vocabulary)
    print(len(vocabulary))