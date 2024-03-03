import os
import numpy as np
import pandas as pd
import math
import re

sentences = {
    "Test temp": "Test temperature: {} K.",
    "Solidus temperature" : "Solidus temperature: {} K.",
    "W":"W composition: {}.",
    "Nb":"Nb composition: {}.",
    "Hf":"Hf composition: {}.",
    "Mo":"Mo composition: {}.",
    "Re":"Re composition: {}.",
    "Zr":"Zr composition: {}.",
    "Ta":"Ta composition: {}.",
    "Ti":"Ti composition: {}.",
    "C":"C composition: {}.",
    "Y":"Y composition: {}.",
    "Al":"Al composition: {}.",
    "Si":"Si composition: {}.",
    "V":"V composition: {}.",
}


elements_info = {
    'W': 'Tungsten (Atomic # 74, Weight 183.84, Young\'s Modulus 411 GPa, Microstructure BCC, Melting Point 3422 °C)',
    'Nb': 'Niobium (Atomic # 41, Weight 92.91, Young\'s Modulus 105 GPa, Microstructure BCC, Melting Point 2468 °C)',
    'Hf': 'Hafnium (Atomic # 72, Weight 178.49, Young\'s Modulus 78 GPa, Microstructure HCP, Melting Point 2233 °C)',
    'Mo': 'Molybdenum (Atomic # 42, Weight 95.95, Young\'s Modulus 329 GPa, Microstructure BCC, Melting Point 2623 °C)',
    'Re': 'Rhenium (Atomic # 75, Weight 186.21, Young\'s Modulus 463 GPa, Microstructure HCP, Melting Point 3186 °C)',
    'Zr': 'Zirconium (Atomic # 40, Weight 91.22, Young\'s Modulus 98 GPa, Microstructure HCP, Melting Point 1855 °C)',
    'Ta': 'Tantalum (Atomic # 73, Weight 180.95, Young\'s Modulus 186 GPa, Microstructure BCC, Melting Point 2996 °C)',
    'Ti': 'Titanium (Atomic # 22, Weight 47.87, Young\'s Modulus 116 GPa, Microstructure HCP, Melting Point 1668 °C)',
    'C': 'Carbon (Atomic # 6, Weight 12.01, Young\'s Modulus Varies, Microstructure Varies, Melting Point Sublimation around 3915 °C for diamond)',
    'Y': 'Yttrium (Atomic # 39, Weight 88.91, Young\'s Modulus 63 GPa, Microstructure HCP, Melting Point 1799 °C)',
    'Al': 'Aluminum (Atomic # 13, Weight 26.98, Young\'s Modulus 69 GPa, Microstructure FCC, Melting Point 660.3 °C)',
    'Si': 'Silicon (Atomic # 14, Weight 28.09, Young\'s Modulus 130 GPa, Microstructure Diamond cubic, Melting Point 1414 °C)',
    'V': 'Vanadium (Atomic # 23, Weight 50.94, Young\'s Modulus 128 GPa, Microstructure BCC, Melting Point 1910 °C)'
}

elements_info = {
    'W': 'Tungsten (Atomic # 74, Weight 183.84',
    'Nb': 'Niobium (Atomic # 41, Weight 92.91',
    'Hf': 'Hafnium (Atomic # 72, Weight 178.49',
    'Mo': 'Molybdenum (Atomic # 42, Weight 95.95',
    'Re': 'Rhenium (Atomic # 75, Weight 186.21',
    'Zr': 'Zirconium (Atomic # 40, Weight 91.22',
    'Ta': 'Tantalum (Atomic # 73, Weight 180.95',
    'Ti': 'Titanium (Atomic # 22, Weight 47.87',
    'C': 'Carbon (Atomic # 6, Weight 12.01',
    'Y': 'Yttrium (Atomic # 39, Weight 88.91',
    'Al': 'Aluminum (Atomic # 13, Weight 26.98',
    'Si': 'Silicon (Atomic # 14, Weight 28.09',
    'V': 'Vanadium (Atomic # 23, Weight 50.94'
}

# Update folder path
folder_path = "data/ys_clean"

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        print(file_name)
        file_name_without_extension = os.path.splitext(file_name)[0]
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df_features = df[[col for col in df.columns if col != 'YS' and col != 'Cr']]
        df_features_dict = df_features.to_dict(orient='records')
        all_records_desc = []
        for record in df_features_dict:
            record_desc = ""
            for property in record:
                if property in sentences and pd.isnull(record[property]) == False:
                    record_desc += sentences[property].format(record[property])
                if property in elements_info:
                    record_desc += " " + elements_info[property]
            all_records_desc.append(record_desc)
        df_to_save = df
        df_to_save['desc'] = all_records_desc
        df_to_save = df_to_save[['YS', 'desc']]
        df_to_save = df_to_save.rename(columns={"desc": "text", "YS": "target"})
        df_to_save.to_pickle('ys_clean_3_sets/'+file_name_without_extension+'.pkl')
