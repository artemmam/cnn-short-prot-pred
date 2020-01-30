import Bio
from Bio.PDB import PDBList
import pandas as pd
'''Selecting structures from PDB'''
pdbl = PDBList()
pdb_list = pd.read_csv('./data/pdb_200.csv')
for i in pdb_list['pdb_name']:
    pdbl.retrieve_pdb_file(i, pdir='./data/pdb/')
