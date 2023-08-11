import os
import re
import pandas as pd

path = ""
os.chdir(path)

generic_abbreviations = pd.read_csv("Abbreviations/generic_abbreviations.csv", na_filter = False)
arbo_abbreviations = pd.read_csv("Abbreviations/Arbo_abbreviations.csv", na_filter = False)
academic_abbreviations = pd.read_csv("Abbreviations/academic_abbreviations.csv", na_filter = False)
environmental_abbreviations = pd.read_csv("Abbreviations/environmental_abbreviations.csv", na_filter = False)
government_abbreviations = pd.read_csv("Abbreviations/government_abbreviations.csv", na_filter = False)
ict_abbreviations = pd.read_csv("Abbreviations/ict_abbreviations.csv", na_filter = False)
it_abbreviations = pd.read_csv("Abbreviations/it_abbreviations.csv", na_filter = False)
laws_abbreviations = pd.read_csv("Abbreviations/laws_abbreviations.csv", na_filter = False)
legal_abbreviations = pd.read_csv("Abbreviations/legal_abbreviations.csv", na_filter = False)
medical_abbreviations = pd.read_csv("Abbreviations/medical_abbreviations.csv", na_filter = False)
navigation_abbreviations = pd.read_csv("Abbreviations/navigation_abbreviations.csv", na_filter = False)
polymer_ISO_abbreviations = pd.read_csv("Abbreviations/polymer_iso_abbreviations.csv", na_filter = False)

abbreviations = [generic_abbreviations['Afkorting'], 
                 ict_abbreviations['Afkorting'], 
                 it_abbreviations['Afkorting'],
                 arbo_abbreviations['Afkorting'],
                 academic_abbreviations['Afkorting'],
                 environmental_abbreviations['Afkorting'],
                 government_abbreviations['Afkorting'],
                 laws_abbreviations['Afkorting'],
                 legal_abbreviations['Afkorting'],
                 medical_abbreviations['Afkorting'],
                 navigation_abbreviations['Afkorting'],
                 polymer_ISO_abbreviations['Afkorting']]

result_flat = []

for i in abbreviations:
    result = [re.split(r'\s\w+\s|\s*\/\s*|\;\s*', s) for s in i]
    result = [[re.sub(r'\s*\(.+\)', "", s) for s in sublist] for sublist in result]
    for l in result:
        for item in l:
            result_flat.append(item)
            
abbreviations = [s for s in result_flat if re.match(r"\w*\..*", s)]
abbreviations = [re.sub(r'\[.*\]', "", s) for s in abbreviations]

abbreviations = list(dict.fromkeys(abbreviations))