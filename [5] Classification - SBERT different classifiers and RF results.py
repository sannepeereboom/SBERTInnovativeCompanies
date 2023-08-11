import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

path = ""
os.chdir(path)


########################################################
########################################################
## Classification of sentences using different models ##
########################################################
########################################################


SCR2_sent = pd.read_parquet("Datasets/SCR2-5_sent_embedded.parquet")
train, test = train_test_split(SCR2_sent, test_size = 0.2, random_state = 42)

# Algorithms 
nb = BernoulliNB()
logreg = LogisticRegression(penalty = "l1", solver = "liblinear")
knn = KNeighborsClassifier(n_neighbors = 2)
sgd = SGDClassifier()
qda = QuadraticDiscriminantAnalysis()
nn = MLPClassifier()
tree = DecisionTreeClassifier()
rf = RandomForestClassifier()
gtb = HistGradientBoostingClassifier()


print('Fitting Bernoulli Naive Bayes model.')
nb.fit(train.loc[:, '0':'767'], train['Innov'])
print('Predicting Bernoulli Naive Bayes model.')
nb_pred = nb.predict(test.loc[:, '0':'767'])

print('Fitting Logistic Regression model.')
logreg.fit(train.loc[:, '0':'767'], train['Innov'])
print('Predicting Logistic Regression model.')
logreg_pred = logreg.predict(test.loc[:, '0':'767'])

print('Fitting 2 nearest neighbors model.')
knn.fit(train.loc[:, '0':'767'], train['Innov'])
print('Predicting 2 nearest neighbors model.')
knn_pred = knn.predict(test.loc[:, '0':'767'])

print('Fitting Stochastic Gradient Descent model.')
sgd.fit(train.loc[:, '0':'767'], train['Innov'])
print('Predicting Stochastic Gradient Descent model.')
sgd_pred = sgd.predict(test.loc[:, '0':'767'])

print('Fitting Quadratic Discriminant Analysis model.')
qda.fit(train.loc[:, '0':'767'], train['Innov'])
print('Predicting Quadratic Discriminant Analysis model.')
qda_pred = qda.predict(test.loc[:, '0':'767'])

print('Fitting Neural Network model.')
nn.fit(train.loc[:, '0':'767'], train['Innov'])
print('Predicting Neural Network model.')
nn_pred = nn.predict(test.loc[:, '0':'767'])

print('Fitting Decision Tree model.')
tree.fit(train.loc[:, '0':'767'], train['Innov'])
print('Predicting Decision Tree model.')
tree_pred = tree.predict(test.loc[:, '0':'767'])

print('Fitting Random Forest model.')
rf.fit(train.loc[:, '0':'767'], train['Innov'])
print('Predicting Random Forest model.')
rf_pred = rf.predict(test.loc[:, '0':'767'])

print('Fitting Gradient Tree Boosting model.')
gtb.fit(train.loc[:, '0':'767'], train['Innov'])
print('Predicting Gradient Tree Boosting model.')
gtb_pred = gtb.predict(test.loc[:, '0':'767'])



print('Predictions Bernoulli Naive Bayes model.')
print(classification_report(test['Innov'], nb_pred, labels=[0,1], target_names = ["NINN", "INN"]))
print('Predictions Logistic Regression model.')
print(classification_report(test['Innov'], logreg_pred, labels=[0,1], target_names = ["NINN", "INN"]))
print('Predictions 2NN model.')
print(classification_report(test['Innov'], knn_pred, labels=[0,1], target_names = ["NINN", "INN"]))
print('Predictions Stochastic Gradient Descent model.')
print(classification_report(test['Innov'], sgd_pred, labels=[0,1], target_names = ["NINN", "INN"]))
print('Predictions QDA model.')
print(classification_report(test['Innov'], qda_pred, labels=[0,1], target_names = ["NINN", "INN"]))
print('Predictions Neural Network model.')
print(classification_report(test['Innov'], nn_pred, labels=[0,1], target_names = ["NINN", "INN"]))
print('Predictions Decision Tree model.')
print(classification_report(test['Innov'], tree_pred, labels=[0,1], target_names = ["NINN", "INN"]))
print('Predictions RF model.')
print(classification_report(test['Innov'], rf_pred, labels=[0,1], target_names = ["NINN", "INN"]))
print('Predictions Gradient Tree Boosting model.')
print(classification_report(test['Innov'], gtb_pred, labels=[0,1], target_names = ["NINN", "INN"]))


###########################
###########################
## Hyperparameter tuning ##
###########################
###########################


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(100, 1000, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt', 80, 160, 240]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(100, 1000, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 100]


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_RSCV = RandomizedSearchCV(RandomForestClassifier(),
                             param_distributions = random_grid, 
                             n_iter = 50, 
                             cv = 5, 
                             verbose = 3, 
                             random_state = 42, 
                             n_jobs = -1,
                             return_train_score = True)
# Fit the random search model
rf_RSCV.fit(train.loc[:, "0":"767"], train['Innov'])


##################################################
##################################################
## Classification of companies using best model ##
##################################################
##################################################


rf_prob = rf.predict_proba(test)

# Concatenate predicted labels, predicted probabilities of innovation, and test set (to include text)
proba_lab_df = pd.concat([pd.DataFrame(rf_prob[:,1]).rename(columns = {0:"rf_proba"}), 
                          pd.DataFrame(rf_pred).rename(columns = {0:"rf_pred_lab"}),
                          test.reset_index()], axis = 1)
proba_lab_df.to_parquet('Datasets/proba_lab_df.parquet', index = False)

# Plot sentence probabilities
median = np.median(proba_lab_df['rf_proba'])
### Correctly classified sentences
plt.hist(proba_lab_df[proba_lab_df['Innov'] == proba_lab_df['rf_pred_lab']]['rf_proba'],
         bins=100, 
         range=(0, 1), 
         edgecolor='black', 
         color = 'green',
         label='Correctly predicted label')
### Incorrectly classified sentences
plt.hist(proba_lab_df[proba_lab_df['Innov'] != proba_lab_df['rf_pred_lab']]['rf_proba'],
         bins = 100, 
         range=(0,1),
         edgecolor='black',
         color='red', 
         alpha = 0.5,
         label='Incorrectly predicted label')
plt.xlabel('Predicted Class Probability')
plt.ylabel('Frequency')
### Plot decision boundary == max probability for which label NINN was predicted
plt.axvline(x = proba_lab_df[proba_lab_df['rf_pred_lab'] == 0].max()['rf_proba'],
            color='black', linestyle='--', lw =2, alpha = 1,
           label='Label classification boundary')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', fontsize = 'x-small')
plt.show()

# Test different methods of label prediction at company level
print('Predictions RF model mean of predicted Innov labels > 0.5.')
print(classification_report(proba_lab_df.groupby('BEID')['Innov'].mean(),
                            proba_lab_df.groupby('BEID')['rf_pred_lab'].mean() > 0.5,
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))

print('Predictions RF model mean p(Innov) > 0.5 == Innov.')
print(classification_report(proba_lab_df.groupby('id_NHR')['Innov'].mean(),
                            proba_lab_df.groupby('id_NHR')['rf_proba'].mean() > 0.5,
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))

print('Predictions RF model mean p(Innov) > 0.55 == Innov.') # Best
print(classification_report(proba_lab_df.groupby('id_NHR')['Innov'].mean(),
                            proba_lab_df.groupby('id_NHR')['rf_proba'].mean() > 0.55,
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))

print('Predictions RF model mean p(Innov) > 0.6 == Innov.')
print(classification_report(proba_lab_df.groupby('id_NHR')['Innov'].mean(),
                            proba_lab_df.groupby('id_NHR')['rf_proba'].mean() > 0.6,
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))

# Import, train, classify, and assess full website texts using best algo
SCR2_par = pd.read_csv("Datasets/SCR2-5_par_embedded.csv")
train_par, test_par = train_test_split(SCR2_par, test_size = 0.2, random_state = 42)
del SCR2_par
rf_par = RandomForestClassifier()
rf_par.fit(train_par.loc[:, '0':'767'], train_par['Innov'])
rf_par_pred = rf.predict(test_par.loc[:, '0':'767'])
rf_par_prob =  rf.predict_proba(test_par)

proba_lab_df_par = pd.concat([pd.DataFrame(rf_par_prob[:,1]).rename(columns = {0:"rf_proba"}),
                              pd.DataFrame(rf_par_pred).rename(columns = {0:"rf_pred_lab"}),
                              test_par.reset_index()], axis = 1)

# Test different methods of label prediction at company level
print('Predictions RF model predicted labels.')
print(classification_report(proba_lab_df_par['Innov'],
                            proba_lab_df_par['rf_pred_lab'],
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))

print('Predictions RF model mean p(Innov) > 0.5 == Innov.')
print(classification_report(proba_lab_df_par['Innov'],
                            proba_lab_df_par['rf_proba'] > 0.5,
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))

print('Predictions RF model mean p(Innov) > 0.55 == Innov.') # Best
print(classification_report(proba_lab_df_par['Innov'],
                            proba_lab_df_par['rf_proba'] > 0.55,
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))

print('Predictions RF model mean p(Innov) > 0.6 == Innov.')
print(classification_report(proba_lab_df_par['Innov'],
                            proba_lab_df_par['rf_proba'] > 0.6,
                            labels=[0,1], 
                            target_names = ["NINN", "INN"]))


####################################################################################
####################################################################################
## Train model on entire dataset at t = 3 for CD and external validation analyses ##
####################################################################################
####################################################################################


rf_full = RandomForestClassifier()
print("Fitting...")
rf_full.fit(SCR2_sent.loc[:, '0':'767'], SCR2_sent['Innov'])
print("Done fitting.")

filename = 'Datasets/random_forest_trained_SC2.pickle'
with open(filename, 'wb') as file:
    pickle.dump(rf_full, file)