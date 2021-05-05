from gensim import models, similarities, corpora
import nltk
from nltk.corpus import stopwords 
import os
import ssl
import pandas as pd

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words("english")
#stop_words = set(stopwords.words('english'))

df = pd.read_csv('quotes_likes/quotes_likes_0-100K.csv', header=0)
df = df.head(10000)

# inverted index for tags
df = df[['quote', 'author', 'tags', 'likes']]
df['tags'] = df['tags'].str.split(',')
df['tags'] = df['tags'].apply(lambda l: list(map(lambda element: element.strip(), l)    ) if type(l)==list else l)
df = df[df['tags'].notnull()]

documents = df['quote'].tolist()
df = df[df['tags'].notnull()]
df['tags'] = df['tags'].str.split(',')
df['tags'] = df['tags'].apply(lambda l: list(map(lambda element: element.strip(), l)) if type(l)==list else l)
texts = [
    [word for word in document.lower().split() if word not in stop_words]
    for document in documents
]

dictionary = corpora.Dictionary(texts)
dictionary.save('quotes_likes/quotes.dict')

corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=500)
lsi.save('quotes_likes/quotes.model',pickle_protocol=4, separately=['model'])

index = similarities.MatrixSimilarity(lsi[corpus])
index.save("quotes_likes/quotes.index")

def query_quotes(queries):
  for query in queries:
    doc = [word for word in query.lower().split() if word not in stop_words]
    vec_bow = dictionary.doc2bow(doc)
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    print(type(sims))
    print(type(sims[0]))
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sims[:10])
    print('QUERY: {}'.format(doc))
    for doc_position, doc_score in sims[:10]:
        print(doc_score, documents[doc_position])
    print()

query_quotes([
'My friends and I are drifting apart from each other.'])
