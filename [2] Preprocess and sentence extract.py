import os
import re
import pandas as pd
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters, PunktLanguageVars
from abbreviations import abbreviations # Exported abbrevs from wikipedia
import langdetect

path = ""
os.chdir(path)

# Read files
SCR01 = pd.read_csv('Datasets/SCR01_htmls_NHR_in_CIS_and_SCR05.csv')
SCR05 = pd.read_csv('Datasets/SCR05_htmls_EM_in_CIS_and_SCR01.csv')
ABR_sample = pd.read_csv('Datasets/NHR_htmls_40000_sample.csv')

SCR01 = SCR01.rename(columns = {"text1":"text"})
SCR05 = SCR05.rename(columns = {"text5":"text"})


# Standardize phone numbers. Note: This apparently also catches some dates and other numbers.
phone_numbers = re.compile('[\+\(\d-]?[1-9][0-9 .\-\(\)]{7,}[0-9]')
# Standardize emails
emails = re.compile('\w+[\.-]*\w*@{1}\w+[\.-]*\w*\.{1}\w+[\.-]*\w*')
# Add space after sentence delimiter if parsing is funky
missing_space = re.compile('([A-Za-z]+[\.\!\?]+)([A-Z]{1}[A-Za-z]*)')
# Add period between end of sentence and start of new sentence if parsing is funky
missing_period = re.compile('(\s[A-Z]?[a-z]{2,})([\.{3,}]*[\(.\)]*)([A-Z]{1}[a-z]+)')

def preprocess(datasets):
    for i in datasets:
        i['text'] = i['text'].apply(lambda x: re.sub(phone_numbers, "phonenumber", str(x)))
        i['text'] = i['text'].apply(lambda x: re.sub(emails, "email", str(x)))
        i['text'] = i['text'].apply(lambda x: re.sub(missing_space, r'\1 \2', str(x)))
        i['text'] = i['text'].apply(lambda x: re.sub(missing_period, r'\1. \3', str(x)))

preprocess([SCR01, SCR05, ABR_sample])


# Export full website text (stored as paragraph) to be passed to embedder
SCR01.to_csv("Datasets/proc_SCR01_htmls_NHR_in_CIS_and_SCR05.csv")
SCR05.to_csv("Datasets/proc_SCR05_htmls_EM_in_CIS_and_SCR01.csv")
ABR_sample.to_csv("Datasets/proc_NHR_htmls_40000_sample.csv")


#########################
#########################
## Sentence extraction ##
#########################
#########################


# Define tokenizer and pass Dutch abbreviations to tokenizer
def sentence_tokenizer(text, lang):
    class endchars(PunktLanguageVars):
        sent_end_chars = ('|', '>>', '.', '?', '!', "...", "..")
    
    sentence_tokenizer =  nltk.data.load(f"tokenizers/punkt/{lang}.pickle")
    sentence_tokenizer._params.abbrev_types.update(abbreviations)
    sentence_tokenizer._lang_vars = endchars()
    
    return sentence_tokenizer.tokenize(text)

def detect_language(text):
    try:
        text_language = langdetect.detect(text)
    except:
        text_language = ""              
    if text_language=="nl":
        text_language="dutch"
    elif text_language=="en":
        text_language="english"
    else:
        text_language="english"
    return text_language

# Initialize dataframes
ABR_sentence_df_text = pd.DataFrame(columns = ["text", "Innov", "id"])
sentence_df_text1 = pd.DataFrame(columns = ["text", "Innov", "id_NHR"])
sentence_df_text5 = pd.DataFrame(columns = ["text","Innov", "BEID"])

# Extract separate sentences from df and store Innov for each sentence
for df in [SCR01, SCR05, ABR_sample]:
    for i in df.index:
        Innov = df.loc[i, 'Innov']

        text = df.loc[i, 'text']
        lang = detect_language(text)
        text = pd.DataFrame(sentence_tokenizer(text, lang), columns = ['text'])
        text_Innov = pd.DataFrame(pd.Series(Innov).repeat(len(text)).T, columns = ['Innov']).reset_index(drop=True)
        text = pd.concat([text, text_Innov], axis = 1, ignore_index = False)

    if df.equals(SCR01):
        SCR01_id = df.loc[i, 'id_NHR']
        SCR01_id = pd.DataFrame(pd.Series(SCR01_id).repeat(len(text)).T, columns = ['id_NHR']).reset_index(drop=True)
        text = pd.concat([text, SCR01_id], axis = 1, ignore_index = False)
        sentence_df_text1 = pd.concat([sentence_df_text1, text], axis = 0, ignore_index = True)
    elif df.equals(SCR05):
        SCR05_id = df.loc[i, 'BEID']
        SCR05_id = pd.DataFrame(pd.Series(SCR05_id).repeat(len(text)).T, columns = ['BEID']).reset_index(drop=True)
        text = pd.concat([text, SCR05_id], axis = 1, ignore_index = False)
        sentence_df_text5 = pd.concat([sentence_df_text5, text], axis = 0, ignore_index = True) 
    elif df.equals(ABR_sample):
        ABR_id = df.loc[i, 'id']
        ABR_id = pd.DataFrame(pd.Series(ABR_id).repeat(len(text)).T, columns = ['id']).reset_index(drop=True)
        text = pd.concat([text, ABR_id], axis = 1, ignore_index = False)
        ABR_sentence_df_text = pd.concat([ABR_sentence_df_text, text], axis = 0, ignore_index = True)  

# Save separate sentences to be passed to embedder
sentence_df_text1.to_csv("Datasets/sentence_df_text1.csv")
sentence_df_text5.to_csv("Datasets/sentence_df_text5.csv")
ABR_sentence_df_text.to_parquet("Datasets/ABR_sentence_df_text.parquet")

