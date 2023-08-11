import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = ""
os.chdir(path)


#####################
#####################
## Import datasets ##
#####################
#####################


datasets = {"SCR01": "1_InnovatiefSurveySDS.csv",
            "SCR05": "5_InnovatieSurvey2_2b_concp15.csv"}
SCR01 = pd.read_csv('Datasets/' + datasets["SCR01"], sep=";")
SCR01 = SCR01.fillna(" ")
SCR05 = pd.read_csv('Datasets/' + datasets["SCR05"], sep=";")
SCR05 = SCR05.fillna(" ")


######################
######################
## Combine datasets ##
######################
######################


SCR01_text = pd.DataFrame()

# Read SCR01 files by SCR05 BEID
for i in SCR05['BEID']:
    txt = str(i) + ".txt"
    with open(txt) as f:
        lines = f.readlines()
    count = len(lines)
    lines = ''.join(lines)
    lines = lines.replace("\n", " ")
    SCR01_text = pd.concat([SCR01_text, pd.DataFrame([i, lines, count]).T], axis = 0,ignore_index=True)
        
SCR01_text.rename({0: "BEID", 1:"SCR01_text", 2:"SCR01_count"}, axis = 1, inplace = True)

SCR_01_05 = pd.merge(SCR01_text, SCR05, on = "BEID")
SCR_01_05.rename({"text":"SCR05_text", "count":"SCR05_count"}, axis = 1, inplace = True)
SCR_01_05.to_csv("SCR_01_05.csv", ";")

# Select ids for which full texts available at both timepoints
url_match = pd.read_csv('Datasets/url_matching.csv')
SCR_01_05 = SCR_01_05[SCR_01_05['BEID'].isin(url_match['BEID'])]
 

###############################
###############################
## Compare text similarities ##
###############################
###############################


# Group by BEID
grouped = SCR_01_05.groupby(["BEID"])
similarity_df = pd.DataFrame(columns = ["BEID", "Cosine similarity", "Proportion similarity"])

# Iterate over BEIDS
for BEID, group in grouped:
    # Extract the SCR01_text and SCR05_text columns
    text1 = group["SCR01_text"].iloc[0]
    text2 = group["SCR05_text"].iloc[0]
    
    # Calculate proportion similarity
    set1 = set(text1.split())
    set2 = set(text2.split())
    unique_words_1 = set1.difference(set2)
    unique_words_2 = set2.difference(set1)
    common_words = set1.intersection(set2)
    proportion_sim = len(common_words) / (len(unique_words_1) + len(unique_words_2) + len(common_words))
    
    # Append similarity measures to df
    new_row = {"BEID": BEID, 
               "Innov": group["Innov"].iloc[0], 
               "Proportion similarity": proportion_sim}
    similarity_df = pd.concat([similarity_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)


#################################################
#################################################
## Generate plots for similarity distributions ##
#################################################
#################################################


# Proportion similarity plot
plt.hist(similarity_df["Proportion similarity"], edgecolor='black', color = "gray", bins = 100)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Frequency")
plt.xlabel("Proportion of common words to unique words")
plt.axvline(similarity_df['Proportion similarity'].median(), color = 'red', ls = "--")
plt.tight_layout()
plt.show()

# Median similarity of website texts
print("Proportion similarity mean", similarity_df['Proportion similarity'].mean())
    
