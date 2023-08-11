import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

path = ""
os.chdir(path)

embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


#############################
#############################
## Embedding dataset t = 3 ##
#############################
#############################


# Full texts

paragraphs = pd.read_csv('Datasets/proc_SCR01_htmls_NHR_in_CIS_and_SCR05.csv')
par_embeddings = embedder.encode(paragraphs['text1'])
pd.concat((paragraphs, pd.DataFrame(par_embeddings)), axis = 1).to_csv("Datasets/SCR2-5_par_embedded.csv", index = False)
del paragraphs, par_embeddings

# Separate sentences

sentences = pd.read_csv('Datasets/sentence_df_text1.csv')
sentences = sentences[~sentences['text'].str.contains('^[^a-zA-z]+$')]

# Cycles because it's a large df
cycles = range(1, np.ceil(sentences.shape[0]/10000).astype(int))
print("Embedding in ", len(cycles)+1, " chunks.")
print('Running cycle 1 ...')
sent_embeddings = embedder.encode(sentences.loc[:9999]['text'])

for i in cycles:
    print('Running cycle', i+1, "...")
    if i != max(cycles):
        left = i * 10000
        right = (i + 1) * 10000 - 1
        embedding_chunk = embedder.encode(list(sentences.loc[left:right]['text']))
        sent_embeddings = np.concatenate((sent_embeddings, embedding_chunk))
        del embedding_chunk
    else:
        left = i * 10000
        embedding_chunk = embedder.encode(list(sentences.loc[left:]['text']))
        sent_embeddings = np.concatenate((sent_embeddings, embedding_chunk))
        del embedding_chunk

pd.concat((sentences, pd.DataFrame(sent_embeddings)), axis = 1).to_parquet("Datasets/SCR2-5_sent_embedded.parquet", index = False)
del cycles, i, left, right, sentences, sent_embeddings


##############################
##############################
## Embedding dataset t = 12 ##
##############################
##############################


# Full texts

paragraphs = pd.read_csv('Datasets/proc_SCR05_htmls_EM_in_CIS_and_SCR01.csv')
par_embeddings = embedder.encode(paragraphs['text5'])
pd.concat((paragraphs, pd.DataFrame(par_embeddings)), axis = 1).to_csv("Datasets/SCR5-2_par_embedded.csv", index = False)
del paragraphs, par_embeddings


# Separate sentences

sentences = pd.read_csv('Datasets/sentence_df_text5.csv')
sentences = sentences[~sentences['text'].str.contains('^[^a-zA-z]+$')]

# Cycles because it's a large df
cycles = range(1, np.ceil(sentences.shape[0]/10000).astype(int))
print("Embedding in ", len(cycles)+1, " chunks.")
print('Running cycle 1 ...')
sent_embeddings = embedder.encode(sentences.loc[:9999]['text'])

for i in cycles:
    print('Running cycle', i+1, "...")
    if i != max(cycles):
        left = i * 10000
        right = (i + 1) * 10000 - 1
        embedding_chunk = embedder.encode(list(sentences.loc[left:right]['text']))
        sent_embeddings = np.concatenate((sent_embeddings, embedding_chunk))
        del embedding_chunk
    else:
        left = i * 10000
        embedding_chunk = embedder.encode(list(sentences.loc[left:]['text']))
        sent_embeddings = np.concatenate((sent_embeddings, embedding_chunk))
        del embedding_chunk

pd.concat((sentences, pd.DataFrame(sent_embeddings)), axis = 1).to_parquet("Datasets/SCR5-2_sent_embedded.parquet", index = False)
del cycles, i, left, right, sentences, sent_embeddings 


#############################################
#############################################
## Embedding external validation set t = 3 ##
#############################################
#############################################


# Full texts

paragraphs = pd.read_csv('Datasets/NHR_htmls_40000_sample.csv')
ABR = pd.read_csv('Datasets/2_ABR_adj.csv', sep=";")
paragraphs = pd.merge(paragraphs, ABR, left_on = 'id', right_on = 'BEID')
paragraphs.drop(columns = ["Unnamed: 0", "id", "Innov_x", "text_y", "lang"], inplace = True)
paragraphs.rename(columns = {"text_x":"text", "Innov_y":"Innov"}, inplace = True)

par_embeddings = embedder.encode(paragraphs['text'])
pd.concat((paragraphs, pd.DataFrame(par_embeddings)), axis = 1).to_parquet("Datasets/ABR_par_embedded.parquet", index = False)
del paragraphs, par_embeddings


# Separate sentences

sentences = pd.read_parquet('Datasets/ABR_sentence_df_text.parquet')
sentences = sentences[~sentences['text'].str.contains('^[^a-zA-z]+$')]

# Cycles because it's a VERY large df
cycles = range(1, np.ceil(sentences.shape[0]/10000).astype(int))
print("Embedding in ", len(cycles)+1, " chunks.")
print('Running cycle 1 ...')
sent_embeddings = embedder.encode(sentences.loc[:9999]['text'])


for i in cycles:
    print('Running cycle', i+1, "...")
    if i != max(cycles):
        left = i * 10000
        right = (i + 1) * 10000 - 1
        embedding_chunk = embedder.encode(list(sentences.loc[left:right]['text']))
        sent_embeddings = np.concatenate((sent_embeddings, embedding_chunk))
        del embedding_chunk
    else:
        left = i * 10000
        embedding_chunk = embedder.encode(list(sentences.loc[left:]['text']))
        sent_embeddings = np.concatenate((sent_embeddings, embedding_chunk))
        del embedding_chunk

pd.concat((sentences, pd.DataFrame(sent_embeddings)), axis = 1).to_parquet("Datasets/ABR_sent_embedded.parquet", index = False)
del cycles, i, left, right, sentences, sent_embeddings