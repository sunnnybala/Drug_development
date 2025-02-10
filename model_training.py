
import torch
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import rdChemReactions
from rdkit.Chem import AllChem
from rdkit.Chem import Draw,BRICS,Recap
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdmolops
from rdkit.Chem.BRICS import BRICSDecompose
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols


from PIL import Image

import sys
import re
import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sascorer

df = pd.read_excel('Oxazolidinone.xlsx')
df.drop([0],inplace = True)
df.drop({'Unnamed: 0'},axis = 1, inplace = True)
df.rename({'Unnamed: 3':'MIC'},inplace=True,axis = 1)
df.dropna(inplace = True)
df['REWARD'] = 1/df['MIC']
df['PMIC']=np.log(df['MIC'])
df.reset_index(inplace = True,drop = True)



print("script ran")