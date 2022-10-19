"""## Initialsing

Run all cells in this section to initialise the data in a root folder

root_folder Location : gdrive/MyDrive/FYP/Data/DRP/root_folder/

root_folder contains branches names for example : "root_005"

Last three character is the index of the root.


A new branch name might have to be created. The below code will suggest an index.

The final data will be stored in branch_folder. For example

branch_folder : gdrive/MyDrive/FYP/Data/DRP/root_folder/root_005
"""

import utils_preproc
import utils_data
# import models_deprecated 
from utils_data import *
from utils_preproc import load_drug_smile_X, save_mix_drug_cell_matrix_X, save_mix_drug_geneexpr_matrix_X
import sys

branch = sys.argv[1]

dataset_X = "GDSC"
branch_name = "root_" + branch
root_folder = "root_folder/"

check_duplicate = False
# check_duplicate = True

if (check_duplicate):
  if (branch_name in os.listdir(root_folder)):
    new_fol_str = str(max([int(fol[-3:]) for fol in os.listdir(root_folder)]) + 1)
    print(f"new_fol_str = {new_fol_str}")
    while (len(new_fol_str) < 3):
      new_fol_str = "0" + new_fol_str
    new_fol_str = "root_" + new_fol_str
    print(f"root_folder = {root_folder}")
    raise ValueError(f'{branch_name} already exists in the folder {root_folder}. Try naming the folder : {new_fol_str}')

branch_folder = root_folder + branch_name
# branch_folder
os.makedirs(branch_folder, exist_ok=True)

do_ordinary_atom_feat = True
# do_ordinary_atom_feat = False
do_mol_ecfp = False
fpl = None
do_edge_features = True
# do_atom_ecfp = False
do_atom_ecfp = True
ecfp_radius = 4   # this is actually radius + 1, set it to 4 for ECFP6  (256 features)
use_radius = None
use_relational_edge = False 
# only applicable to gene expression data, not CNV data
top_n_gene = 4000   # use top_n_gene = None to involve all genes 
assert top_n_gene <= 13142

drug_dict_X, drug_smile_X, smile_graph_X = load_drug_smile_X(do_ordinary_atom_feat, do_mol_ecfp, fpl, do_edge_features, do_atom_ecfp, ecfp_radius, use_radius, use_relational_edge)
# xd_X, xc_X, y_X, dgl, cosl = save_mix_drug_cell_matrix_X(do_ordinary_atom_feat, do_mol_ecfp, fpl, do_edge_features, do_atom_ecfp, ecfp_radius, use_radius)
xd_X, xc_X, y_X, dgl, cosl = save_mix_drug_geneexpr_matrix_X(do_ordinary_atom_feat, do_mol_ecfp, fpl, do_edge_features, do_atom_ecfp, ecfp_radius, use_radius, use_relational_edge, top_n_gene)

# print(smile_graph_X)

# for blind test (drugs appearing in the testing set do not appear in the training set), set randomize = False
randomize = True
seed = 19871729 ## start from 19871729, add one each time for multiple testing

if (randomize):
    np.random.seed(seed)
    np.random.shuffle(xd_X)

    np.random.seed(seed)
    np.random.shuffle(xc_X)

    np.random.seed(seed)
    np.random.shuffle(y_X)
    
    np.random.seed(seed)
    np.random.shuffle(dgl)
    
    np.random.seed(seed)
    np.random.shuffle(cosl)

size_X = int(xd_X.shape[0] * 0.8)
size1_X = int(xd_X.shape[0] * 0.9)

xd_train_X = xd_X[:size_X]
print('xd_train_X',xd_train_X.shape)
xd_test_X = xd_X[size_X:size1_X]
xd_val_X = xd_X[size1_X:]

xc_train_X = xc_X[:size_X]
xc_test_X = xc_X[size_X:size1_X]
xc_val_X = xc_X[size1_X:]

y_train_X = y_X[:size_X]
y_test_X = y_X[size_X:size1_X]
y_val_X = y_X[size1_X:]

dgl_train_X = dgl[:size_X]
dgl_test_X = dgl[size_X:size1_X]
dgl_val_X = dgl[size1_X:]

cosl_train_X = cosl[:size_X]
cosl_test_X = cosl[size_X:size1_X]
cosl_val_X = cosl[size1_X:]

print(f"xd_X.shape[0] = {xd_X.shape[0]}")
print(f"size_X = {size_X}")
print(f"size1_X = {size1_X}")
print()
print(f"xd_train_X = {xd_train_X.shape}")
print(f"xd_val_X = {xd_val_X.shape}")
print(f"xd_test_X = {xd_test_X.shape}")
print()
print(f"xc_train_X = {xc_train_X.shape}")
print(f"xc_val_X = {xc_val_X.shape}")
print(f"xc_test_X = {xc_test_X.shape}")
print()
print(f"y_train_X = {y_train_X.shape}")
print(f"y_val_X = {y_val_X.shape}")
print(f"y_test_X = {y_test_X.shape}")
print()
print(f"dgl_train_X = {dgl_train_X.shape}")
print(f"dgl_test_X = {dgl_test_X.shape}")
print(f"dgl_val_X = {dgl_val_X.shape}")
print()
print(f"cosl_train_X = {cosl_train_X.shape}")
print(f"cosl_test_X = {cosl_test_X.shape}")
print(f"cosl_val_X = {cosl_val_X.shape}")


dataset_X = 'GDSC'
print('preparing ', dataset_X + '_train.pt in pytorch format!')                         ##

print(f"root_folder = {root_folder}")
print(f"branch_name = {branch_name}")
print(f"branch_folder = {branch_folder}")

this_branch = branch_name
train_data = TestbedDataset(root='root_folder/'+ this_branch, dataset=dataset_X+'_train_mix', xd=xd_train_X, xt=xc_train_X, y=y_train_X, smile_graph=smile_graph_X, dgl=dgl_train_X, cosl = cosl_train_X)
val_data = TestbedDataset(root='root_folder/' + this_branch, dataset=dataset_X+'_val_mix', xd=xd_val_X, xt=xc_val_X, y=y_val_X, smile_graph=smile_graph_X, dgl=dgl_val_X, cosl = cosl_val_X)
test_data = TestbedDataset(root='root_folder/'+ this_branch, dataset=dataset_X+'_test_mix', xd=xd_test_X, xt=xc_test_X, y=y_test_X, smile_graph=smile_graph_X, dgl=dgl_test_X, cosl=cosl_test_X)
