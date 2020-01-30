from Bio.PDB import *
import numpy as np
import math
from glob import glob
import pandas as pd
import pickle

parser = PDBParser()


def two_maps(res_list):
    """
       Create matrix of aminoacids distances for all proteins in pdb_200.csv (some precalculated file with "clean"
       proteins and legth<=200)
       :param res_list:
       :return:
       """
    coords = []
    S = 0
    for res in res_list:
        if str(res)[8:12] == 'HOH':
            break
        else:
            for atoms in res:
                if str(atoms) == '<Atom CA>':
                    #print(str(atoms))
                    #print(atoms.get_coord())
                    coords.append(atoms.get_coord())
                    S +=1
                #for j in range(3):
                   # s[i] = atoms.get_vector()
    coords = np.array(coords)
    map_of_cont = np.zeros((len(coords), len(coords)))
    map_of_dist = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)-1):
        for j in range (i+1, len(coords)):
            #print(i, j)
            map_of_dist[i, j] = math.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2 + (coords[i, 2] - coords[j, 2])**2)
            map_of_dist[j, i] = map_of_dist[i, j]
            if math.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2 + (coords[i, 2] - coords[j, 2])**2) <= 8:
                map_of_cont[i,j] = 1
                map_of_cont[j,i] = 1
    #print(S)
    return(map_of_dist, map_of_cont)


prot_df = pd.read_csv('./data/pdb_200.csv')
print(len(prot_df))
two_matrix = []
S = 0
#  You need to download
pdb_files = glob('./data/pdb/*.ent')
for fileName in pdb_files:
    structure_id = fileName[-8:-4]
    if structure_id.upper() in prot_df.pdb_name.tolist():
        try:
            structure = parser.get_structure(structure_id, fileName)
            model = structure[0]
            res_list = Selection.unfold_entities(model, 'R')
            A, B = two_maps(res_list)
            two_matrix.append([structure_id.upper(), A, B])
            S += 1
        except PDBConstructionException:
            print('Ошибка!')
    print(S)


output = open('two_matrix_200.pkl', 'wb')
pickle.dump(two_matrix, output)
output.close()