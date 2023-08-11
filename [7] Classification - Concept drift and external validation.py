import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report

path = ""
os.chdir(path)

# Import model trained on full dataset at t = 3
with open('random_forest_trained_SC2.pickle', 'rb') as file:
    rf = pickle.load(file)


#########################
#########################
## Concept drift check ##
#########################
#########################


# Import data at t = 12
SCR5_sent = pd.read_parquet("Datasets/SCR5-2_sent_embedded.parquet")

# Predict
print("Predicting class probabilities and labels...")
probas_5 = rf.predict_proba(SCR5_sent.loc[:, '0':'767'])
labels_5 = rf.predict(SCR5_sent.loc[:, '0':'767'])
proba_lab_df_5 = pd.concat([pd.DataFrame(probas_5[:,1]).rename(columns = {0:"proba"}), 
                            pd.DataFrame(labels_5).rename(columns = {0:"lab"}),
                            SCR5_sent.reset_index()], axis = 1)
# Assess performance
print("Classification report company level at t = 12")
print(classification_report(proba_lab_df_5.groupby('BEID').mean()['Innov'],
                            proba_lab_df_5.groupby('BEID').mean()['proba']> 0.55))


#########################
## External validation ##
#########################

ABR_sent = pd.read_parquet("Datasets/ABR_sent_embedded.parquet")
SCR2_sent = pd.read_parquet("Datasets/SCR2-5_sent_embedded.parquet")
ABR_sent = ABR_sent[~ABR_sent['id'].isin(SCR2_sent['id_NHR'])] # remove ids that were in training data
del SCR2_sent

labels_ABR = rf.predict(ABR_sent.loc[:, '0':'767'])
probas_ABR = rf.predict_proba(ABR_sent.loc[:, '0':'767'])

proba_lab_df_ABR = pd.concat([pd.DataFrame(probas_ABR[:,1]).rename(columns = {0:"proba"}), 
                              pd.DataFrame(labels_ABR).rename(columns = {0:"lab"}),
                              ABR_sent.reset_index()[['text', 'Innov', 'id']]], 
                             axis = 1)

print('Predictions at sentence level.')
print(classification_report(proba_lab_df_ABR['Innov'],
                            proba_lab_df_ABR['lab'],
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))

print('Predictions at company level.')
print(classification_report(proba_lab_df_ABR.groupby('id')['Innov'].mean(),
                            proba_lab_df_ABR.groupby('id')['proba'].mean() > 0.55,
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))