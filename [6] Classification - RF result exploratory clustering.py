import os
import pandas as pd
import numpy as np

# Clustering
from nltk.cluster import KMeansClusterer
import nltk
from sklearn.metrics.pairwise import cosine_distances


path = ""
os.chdir(path)
df = pd.read_parquet('Datasets/proba_lab_df.parquet')


############
############
## Import ##
############
############

# Select relevant columns
df = df.loc[:, ['id_NHR', 
                'text', 
                'Innov', 
                'rf_proba', 
                'rf_pred_lab'] + 
                list(df.loc[:, '0':'767'].columns)]
# Convert embedding vector to list
df['emb'] = df.loc[:, '0':'767'].values.tolist()

# Separate low-middle-high sentence probabilities for clustering
## High-prob sentences
### Correctly classified
inn_df = df.loc[(np.logical_and(df['Innov'] == 1,
                                df['rf_pred_lab'] == 1)) &
                (df['rf_proba'] >= 0.75)].reset_index(drop = True)
### Misclassified
misclass_high_prob = df.loc[(np.logical_and(df['Innov'] == 0,
                                            df['rf_pred_lab'] == 1)) &
                            (df['rf_proba'] >= 0.75)].reset_index(drop = True)
## Low-prob sentences
### Correctly classified
ninn_df = df.loc[(np.logical_and(df['Innov'] == 0, 
                                 df['rf_pred_lab'] == 0)) &
                 (df['rf_proba'] <= 0.35)].reset_index(drop = True)
### Misclassified
misclass_low_prob = df.loc[(np.logical_and(df['Innov'] == 1,
                            df['rf_pred_lab'] == 0)) &
                           (df['rf_proba'] <= 0.35)].reset_index(drop = True)
## Ambiguous sentences
mid_df = df.loc[np.logical_and(df['rf_proba'] > 0.35,
                               df['rf_proba'] < .75)].reset_index(drop = True)


#####################################
#####################################
## Define functions for clustering ##
#####################################
#####################################


def sentence_clustering(data, k):
    X = np.array(data['emb'].tolist())
    kclusterer = KMeansClusterer(k, 
                                 distance = nltk.cluster.util.cosine_distance,
                                 repeats = 10,
                                 avoid_empty_clusters = True)
    assigned_clusters = kclusterer.cluster(X, 
                                           assign_clusters=True)
    data['cluster'] = pd.Series(assigned_clusters, 
                                index = data.index)
    data['centroid'] = data['cluster'].apply(lambda x: kclusterer.means()[x])
    
def centroid_distance(row):
    return cosine_distances(np.array(row['emb']).reshape(1, -1), 
                            np.array(row['centroid']).reshape(1, -1))

print("Clustering correctly classified innovative sentences...")
sentence_clustering(inn_df, k = 20)
print("Calculating distances from centroid...")
inn_df['distance_from_centroid'] = inn_df.apply(centroid_distance,
                                                axis = 1)
print("Clustering incorrectly classified innovative sentences...")
sentence_clustering(misclass_high_prob, k = 10)
print("Calculating distances from centroid...")
misclass_high_prob['distance_from_centroid'] = misclass_high_prob.apply(centroid_distance,
                                                                        axis = 1)

print("Clustering correctly classified Ninnovative sentences...")
sentence_clustering(ninn_df, k = 20)
print("Calculating distances from centroid...")
ninn_df['distance_from_centroid'] = ninn_df.apply(centroid_distance,
                                                  axis = 1)
print("Clustering incorrectly classified Ninnovative sentences...")
sentence_clustering(misclass_low_prob, k = 10)
print("Calculating distances from centroid...")
misclass_low_prob['distance_from_centroid'] = misclass_low_prob.apply(centroid_distance,
                                                                      axis = 1)

print("Clustering middle region sentences...")
sentence_clustering(mid_df, k = 20)
print("Calculating distances from centroid...")
mid_df['distance_from_centroid'] = mid_df.apply(centroid_distance,
                                                axis = 1)


########################################################################
########################################################################
## Manually look at "typical" innovative and non-innovative sentences ##
########################################################################
########################################################################


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)

print(" Innovative sentences - correctly classified")
for i in range(20):
    print(i+1, ".\n\n ", 
          "TOP:\n",
          inn_df[inn_df['cluster'] == i].sort_values('distance_from_centroid').head(50)['text'],
          "\n\nBOT:\n", 
          inn_df[inn_df['cluster'] == i].sort_values('distance_from_centroid').tail(50)['text'],
          "\n\n\n", sep = "")
    

print(" NInnovative sentences - correctly classified")
for i in range(20):
    print(i+1, ".\n\n ", 
          "TOP:\n",
          ninn_df[ninn_df['cluster'] == i].sort_values('distance_from_centroid').head(50)['text'],
          "\n\nBOT:\n", 
          ninn_df[ninn_df['cluster'] == i].sort_values('distance_from_centroid').tail(50)['text'],
          "\n\n\n", sep = "")