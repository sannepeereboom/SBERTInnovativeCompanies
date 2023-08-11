import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
path = ""
os.chdir(path)


# Data import
SCR2_sent = pd.read_parquet("Datasets/SCR2-5_sent_embedded.parquet")
SCR2_sent = SCR2_sent[~SCR2_sent['text'].str.contains('^[^a-zA-z]+$')]

SCR5_sent = pd.read_csv("Datasets/SCR5-2_sent_embedded.csv")
SCR5_sent = SCR5_sent[~SCR5_sent['text'].str.contains('^[^a-zA-z]+$')]


# Url matching file contains companies scraped at both timepoints
# Scrape 2 has 'id_NHR' as ID while scrape 5 has 'BEID' as id
# Url matching file has these IDs matched
# There are some duplicate BEIDs as a result of different id_NHR (but same company url)
id_match = pd.read_csv("Datasets/url_matching.csv")
id_match = id_match[~id_match['BEID'].duplicated()]

# Select IDs from sentence file scrape 2 only from companies in both scrapes
SC2_merged_ids = pd.merge(id_match[['id_NHR', 'BEID']], SCR2_sent, on = 'id_NHR')
# Select from scrape 5 companies that are in scrape 2
SC5_in_SC2 = SCR5_sent[SCR5_sent['BEID'].isin(SC2_merged_ids['BEID'])]
# Select from scrape 2 companies that are in scrape 5
SC2_in_SC5 = SC2_merged_ids[SC2_merged_ids['BEID'].isin(SC5_in_SC2['BEID'])]



print("Companies in both datasets:", 
      len(SC2_merged_ids[SC2_merged_ids['BEID'].isin(SCR5_sent['BEID'])]['BEID'].unique()))
print("Unique companies in scrape 2:",
      len(SC2_merged_ids[~SC2_merged_ids['BEID'].isin(SCR5_sent['BEID'])]['BEID'].unique()))
print("Unique companies in scrape 5:",
      len(SCR5_sent[~SCR5_sent['BEID'].isin(SC2_merged_ids['BEID'])]['BEID'].unique()))

# Import embedded full texts to double-check cosine similarity
par_SC2 = pd.read_csv("Datasets/SCR2-5_par_embedded.csv")
par_SC5 = pd.read_csv("Datasets/SCR5-2_par_embedded.csv")

# Keep only texts where IDs are in both scrapes
par_SC2 = par_SC2[par_SC2['id_NHR'].isin(SC2_in_SC5['id_NHR'])]
par_SC5 = par_SC5[par_SC5['BEID'].isin(SC5_in_SC2['BEID'])]
# Add BEID to scrape 2 to be able to match to scrape 5
par_SC2 = pd.merge(par_SC2, SC2_in_SC5[['id_NHR', 'BEID']], on = 'id_NHR').drop_duplicates()


##################################
##################################
## Sentence embeddings averaged ##
##################################
##################################


# Calculate average embedding per company for each timepoint
mean_SC2 = SC2_in_SC5.groupby(['BEID']).mean(numeric_only = True)
mean_SC5 = SC5_in_SC2.groupby(['BEID']).mean(numeric_only = True)

# Calculate the cosine similarity between timepoints per company
similarity_scores_SBERT = cosine_similarity(mean_SC2.loc[:, '0':'767'], mean_SC5.loc[:, '0':'767'])

# -> manual inspection of similar/dissimilar cases: averaged vectors are bad quality


##########################
##########################
## Full text embeddings ##
##########################
##########################


# Calculate average embedding per company for each timepoint
mean_SC2_par = par_SC2.groupby(['BEID']).mean(numeric_only = True)
mean_SC5_par = par_SC5.groupby(['BEID']).mean(numeric_only = True)

# Calculate the cosine similarity between timepoints per company
similarity_scores_SBERT_par = cosine_similarity(mean_SC2_par.loc[:, '0':'767'], mean_SC5_par.loc[:, '0':'767'])


############################
############################
## Plots and mean/medians ##
############################
############################


# Median cosine similarity for averaged sentence embeddings and full text embeddings
SBERT_med = np.median(similarity_scores_SBERT.diagonal())
SBERT_par_med = np.median(similarity_scores_SBERT_par.diagonal())
print(SBERT_med, SBERT_par_med)


plt.hist(similarity_scores_SBERT_par.diagonal(),  edgecolor='black', bins = 100)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.title("SBERT Full Website Text Embeddings")
plt.ylabel("Frequency")
plt.xlabel("Cosine Similarity")
plt.xlim(0, 1)
plt.axvline(SBERT_par_med,color='red',ls="--")
plt.show()

# Export plot
#plt.savefig(path+"sim_over_time.jpg", dpi = 400)

