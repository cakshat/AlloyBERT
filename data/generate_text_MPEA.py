import os
import numpy as np
import pandas as pd
import math
import re

sentences = {
    "IDENTIFIER: Reference ID": 'Reference ID is {}.',
    "FORMULA": 'Alloy\'s formula is {}.',
    "PROPERTY: Microstructure": 'Microstructure of alloy is {}.',
    "PROPERTY: Processing method": 'Processing method of the alloy is {}.',
    "PROPERTY: BCC/FCC/other": 'The alloy is {}.',
    "PROPERTY: Grain size ($\mu$m)": 'Alloy\'s grain size is {} microns.',
    "PROPERTY: Exp. Density (g/cm$^3$)": 'Experimental density of the alloy is {} g/cm^3.',
    "PROPERTY: Calculated Density (g/cm$^3$)": 'Calculated density of the alloy is {} g/cm^3.',
    "PROPERTY: HV": 'Hardness of the alloy is {} in Vickers scale.',
    "PROPERTY: Type of test": 'The alloy was tested in {} test.',
    "PROPERTY: Test temperature ($^\circ$C)": 'Alloy as tested at {} Celsius.',
    "PROPERTY: YS (MPa)": 'Yield Strength (YS) of alloy is {}MPa.',
    "PROPERTY: UTS (MPa)": 'Ultimate tensile strength (UTS) of alloy is {}MPa.',
    "PROPERTY: Elongation (%)": 'The alloy has {}% elongation.',
    "PROPERTY: Elongation plastic (%)": 'Plastic elongation of the alloy is {}%.',
    "PROPERTY: Exp. Young modulus (GPa)": 'Experimental Young modulus of the alloy is {}GPa.',
    "PROPERTY: Calculated Young modulus (GPa)": 'Calculated Young modulus of the alloy is {}GPa.',
    "PROPERTY: O content (wppm)": 'Oxygen content of alloy is {} wppm.',
    "PROPERTY: N content (wppm)": 'Nitrogen content of alloy is {} wppm.',
    "PROPERTY: C content (wppm)": 'Carbon content of alloy is {} wppm.',
    "REFERENCE: doi": 'The DOI of the paper is {}.',
    "REFERENCE: year": 'The year of the paper is {}.',
    "REFERENCE: title": 'The title of the paper is {}.',
}

processing_methods = {
    'CAST': "Liquid material poured into a mold, solidifying to form the casting. Used for complex shapes, especially for metals or materials like epoxy, concrete, plaster, and clay.",
    'WROUGHT': "Shaping materials through plastic deformation, including rolling, forging, extrusion, or drawing. Commonly applied to metallic alloys for specific shapes, sizes, and mechanical properties.",
    'POWDER': "Production and shaping of metallic alloys as powders. Utilizes techniques like powder production, consolidation, and sintering in powder metallurgy. Offers versatility for controlled composition and complex geometries.",
    'ANNEAL': "Heat treatment process for alloys. Involves heating to a specific temperature, holding, and controlled cooling. Used to relieve stresses, improve machinability, enhance mechanical properties, and alter microstructure."
}


elements = {
    "Al": "Aluminum, atomic number 13, atomic mass 26.98 u. Density 2.70 g/cm³, melts at 660.3°C. FCC microstructure, Young's Modulus ≈ 70 GPa.",
    "Co": "Cobalt, atomic number 27, atomic mass 58.93 u. Density 8.90 g/cm³, melts at 1495°C. HCP microstructure, Young's Modulus ≈ 211 GPa.",
    "Fe": "Iron, atomic number 26, atomic mass 55.85 u. Density 7.87 g/cm³, melts at 1538°C. BCC microstructure, Young's Modulus ≈ 211 GPa.",
    "Ni": "Nickel, atomic number 28, atomic mass 58.69 u. Density 8.91 g/cm³, melts at 1455°C. FCC microstructure, Young's Modulus ≈ 200 GPa.",
    "Si": "Silicon, atomic number 14, atomic mass 28.09 u. Density 2.33 g/cm³, melts at 1414°C. Diamond cubic structure, Young's Modulus ≈ 130 GPa.",
    "Cr": "Chromium, atomic number 24, atomic mass 51.996 u. Density 7.19 g/cm³, melts at 1907°C. BCC microstructure, Young's Modulus ≈ 279 GPa.",
    "Mn": "Manganese, atomic number 25, atomic mass 54.94 u. Density 7.21 g/cm³, melts at 1246°C. BCC microstructure, Young's Modulus ≈ 198 GPa.",
    "Ti": "Titanium, atomic number 22, atomic mass 47.87 u. Density 4.54 g/cm³, melts at 1668°C. HCP structure, Young's Modulus ≈ 116 GPa.",
    "Cu": "Copper, atomic number 29, atomic mass 63.55 u. Density 8.96 g/cm³, melts at 1085°C. FCC structure, Young's Modulus ≈ 130 GPa.",
    "Mo": "Molybdenum, atomic number 42, atomic mass 95.95 u. Density 10.22 g/cm³, melts at 2623°C. BCC structure, Young's Modulus ≈ 329 GPa.",
    "Nb": "Niobium, atomic number 41, atomic mass 92.91 u. Density 8.57 g/cm³, melts at 2468°C. BCC structure, Young's Modulus ≈ 105 GPa.",
    "V": "Vanadium, atomic number 23, atomic mass 50.94 u. Density 6.11 g/cm³, melts at 1910°C. BCC structure, Young's Modulus ≈ 128 GPa.",
    "Zr": "Zirconium, atomic number 40, atomic mass 91.22 u. Density 6.52 g/cm³, melts at 1852°C. HCP structure, Young's Modulus ≈ 88 GPa.",
    "Sn": "Tin, atomic number 50, atomic mass 118.71 u. Density 7.29 g/cm³, melts at 231.9°C. Tetragonal structure, Young's Modulus ≈ 50 GPa.",
    "Ta": "Tantalum, atomic number 73, atomic mass 180.95 u. Density 16.65 g/cm³, melts at 3290°C. BCC structure, Young's Modulus ≈ 200 GPa.",
    "Hf": "Hafnium, atomic number 72, atomic mass 178.49 u. Density 13.31 g/cm³, melts at 2233°C. HCP structure, Young's Modulus ≈ 78 GPa.",
    "W": "Tungsten, atomic number 74, atomic mass 183.84 u. Density 19.25 g/cm³, melts at 3422°C. BCC structure, Young's Modulus ≈ 411 GPa.",
    "Zn": "Zinc, atomic number 30, atomic mass 65.38 u. Density 7.14 g/cm³, melts at 419.5°C. HCP structure, Young's Modulus ≈ 96 GPa.",
    "Re": "Rhenium, atomic number 75, atomic mass 186.21 u. Density 21.04 g/cm³, melts at 3186°C. HCP structure, Young's Modulus ≈ 463 GPa.",
    "Mg": "Magnesium, atomic number 12, atomic mass 24.31 u. Density 1.74 g/cm³, melts at 650°C. HCP structure, Young's Modulus ≈ 45 GPa.",
    "Pd": "Palladium, atomic number 46, atomic mass 106.42 u. Density 12.02 g/cm³, melts at 1554°C. FCC structure, Young's Modulus ≈ 121 GPa."
}


microstructure = {
    "FCC": "Atoms arranged in a cube with one at each corner and one at the center of each face. Packing density ~74%.",
    "BCC": "Atoms arranged in a cube with one at each corner and one at the center. Packing density than FCC ~68%."
}

# Update folder_path
folder_path = "data/MPEA_numeric/MPEA_numeric.csv"

# To iterate over all the files in the folder

for filename in os.listdir(folder_path):
    print(filename)
    # Remove the .csv extension from the filename
    file_name_without_extension = os.path.splitext(filename)[0]
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)
    df_features = df[[col for col in df.columns if col != "PROPERTY: Calculated Young modulus (GPa)"]]
    df_dict = df_features.to_dict('records')
    all_desc = []
    for record in df_dict:
        desc = ''
        for property in record:
            if property in sentences and pd.isnull(record[property]) == False:
                desc += sentences[property].format(record[property])
                if property == 'FORMULA':
                    s = record[property]
                    floats = re.findall(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', s)
                    floats = [float(f[0]) for f in floats]
                    strings = re.split(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', s)
                    strings = [str.strip() for str in strings if str and not str.replace('.','',1).isdigit()]

                    for i in range(len(strings)):
                        if strings[i] in elements:
                            desc += "Alloy has {} atoms of {}.".format(floats[i], strings[i])
                            desc += elements[strings[i]]

                if property == 'PROPERTY: Processing method':
                    if record[property] in processing_methods:
                        desc += processing_methods[record[property]]

                if property == 'PROPERTY: Microstructure':
                    if record[property] in microstructure:
                        desc += microstructure[record[property]]              
        # print(desc)
        all_desc.append(desc)
    df_to_save = df
    df_to_save['desc'] = all_desc
    df_to_save = df_to_save[['PROPERTY: Calculated Young modulus (GPa)', 'desc']]
    df_to_save = df_to_save.rename(columns={"desc": "text", "PROPERTY: Calculated Young modulus (GPa)": "target"})
    
    df_to_save.to_pickle('MPEA_3_sets/'+file_name_without_extension+'.pkl')
