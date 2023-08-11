import os
import bs4
import re
import os
import numpy as np
from unidecode import unidecode
import pandas as pd
path = ("")
os.chdir(path)


###############
###############
## Functions ##
###############
###############


def createsouplocalfile(thisfile, folder = ""):
    # Create a soup based on the local file (make sure the pathname is before the filename)
    try: 
        soup = bs4.BeautifulSoup(open(path + folder + thisfile, encoding='utf-8').read(), "lxml")
        return(soup)
    except:
        print("not possible to read:", thisfile)
        return 0

def visibletext(soup):
    # Extract content from meta-tag
    meta = soup.find('meta', attrs={'name': 'description'})
    
    # Kill all the scripts and style and return texts
    for script in soup.find_all(["script", "style"]):
        script.extract()    # rip it out
        
    # Find links within <li> tags
    # For some reason it pastes all the list texts together otherwise
    for h in soup.find_all(['li']):
        # search <a> tag inside <li> tag
        a = h.find('a', href = True)
        try:
            # set string attribute to the text found betwen <a> tags
            # add \n for later formatting to "sentence"
            a.string = a.text + "\n"
        except:
            pass

    # Remove weird characters and remaining tags like <iframe>
    text = unidecode(soup.get_text())
    text = re.sub(r'<[^>]+>|{[^}]+}', "", text)
    # Concatenate into one paragraph
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    # Change newline into periods while keeping existing punctuation
    text = re.sub(r'([^!.?])(\n)', "\g<1>. ", text)
    text = re.sub(r'\n', " ", text)
    try:
        # Add period (if needed) and space to meta, ensures it's a "sentence"
        # bit convoluted, checks if string ends w/ sentence delimiter
        if meta["content"][-1] not in (".!?"):
            meta = meta["content"] + ". "
        else:
            meta = meta["content"] + " "
        text = meta + text
    except:
        pass
    return(text)

def processHTMLfile(filename, folder = ""):
    ''' process html file with python  '''
    soup = createsouplocalfile(filename, folder)
    if(soup != 0):
    	text = visibletext(soup)
    else:
        text = ""
    return(text)


###################
###################
## Process HTMLS ##
###################
###################


# Import available labels
Innov_SCR01_SCR05 = pd.read_csv('Datasets/url_matching.csv', usecols = ['BEID', 'id_NHR', 'Innov'], dtype = "object")
Innov_ABR_sample = pd.read_csv("Datasets/OutputNHR.txt", dtype = "object", sep = ";")

# Scrape 2 of companies in CIS-2016 for which we still had htmls
## - Which were also in SCR05
## - Which could be matched to respective ID and urls
## - For which we still had htmls
text_df = pd.DataFrame()
folder = 'HTMLs/Scrape 2'
for website in os.listdir(folder):
        NHR_id = website.strip('.html')
        text = processHTMLfile(website, folder = folder+"/")
        text = (NHR_id, text)
        text_df = pd.concat([text_df, pd.DataFrame([text])], axis = 0, ignore_index = True)
text_df.rename(columns = {0:"id_NHR", 1:"text1"}, inplace = True)
text_df = pd.merge(text_df, Innov_SCR01_SCR05[['id_NHR', 'Innov']], on = "id_NHR", validate = "1:1")
# Post-processing of tags -> didn't seem to fully work within visibletext()
## (at least for the ABR sample)
tmp = text_df.copy()
tmp['text'] = tmp['text'].apply(unidecode)
tmp['text'] = tmp['text'].apply(lambda x: re.sub(r'<+[^>]+>+|{+[^}]+}+', "", x)) # removes remaining tags
tmp['text'].replace("", np.nan, inplace = True)
tmp.dropna(subset = ['text'], inplace = True)
tmp['text'] = tmp['text'].apply(lambda x: x.strip())
tmp.to_csv(path+folder+".csv", index = False)

# Scrape 5 of companies in CIS-2016 
## - Which were also in SCR01
## - Which could be matched to respective ID and urls
## - For which we still had htmls
text_df = pd.DataFrame()
folder = 'HTMLs/Scrape 5'
for website in os.listdir(folder):
        BEID = website.strip('.html')
        text = processHTMLfile(website, folder = folder+"/")
        text = (BEID, text)
        text_df = pd.concat([text_df, pd.DataFrame([text])], axis = 0, ignore_index = True)
text_df.rename(columns = {0:"BEID", 1:"text"}, inplace = True)
text_df = pd.merge(text_df, Innov_SCR01_SCR05[['BEID', 'Innov']].drop_duplicates(subset = ['BEID', 'Innov']), on = "BEID", validate = "1:1")
# Post-processing of tags -> didn't seem to fully work within visibletext()
## (at least for the ABR sample)
tmp = text_df.copy()
tmp['text'] = tmp['text'].apply(unidecode)
tmp['text'] = tmp['text'].apply(lambda x: re.sub(r'<+[^>]+>+|{+[^}]+}+', "", x)) # removes remaining tags
tmp['text'].replace("", np.nan, inplace = True)
tmp.dropna(subset = ['text'], inplace = True)
tmp['text'] = tmp['text'].apply(lambda x: x.strip())
tmp.to_csv(path+folder+".csv", index = False)

# Scrapes of sample of companies in ABR
## - Labels generated by original model
## - Sample of 40.000 out of ~600.000
text_df = pd.DataFrame()
folder = 'HTMLs/ABR_t-3'
for website in os.listdir(folder):
        ABR_id = website.strip('.html')
        text = processHTMLfile(website, folder = folder+"/")
        text = (ABR_id, text)
        text_df = pd.concat([text_df, pd.DataFrame([text])], axis = 0, ignore_index = True)
text_df.rename(columns = {0:"id", 1:"text"}, inplace = True)
tmp = pd.merge(text_df, Innov_ABR_sample, on = "id", validate = "1:1")
# Post-processing of tags -> didn't seem to fully work within visibletext()
## (at least for the ABR sample)
tmp['text'] = tmp['text'].apply(unidecode)
tmp['text'] = tmp['text'].apply(lambda x: re.sub(r'<+[^>]+>+|{+[^}]+}+', "", x)) # removes remaining tags
tmp['text'].replace("", np.nan, inplace = True)
tmp.dropna(subset = ['text'], inplace = True)
tmp['text'] = tmp['text'].apply(lambda x: x.strip())
tmp.to_csv(path+folder+".csv", index = False)

