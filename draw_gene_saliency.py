import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import os, argparse
from utils_preproc import preproc_gene_expr, save_gene_expr_matrix_X
from utils_decoding import normalize_ss, rank_ss, draw_one_drug, draw_gene_saliency, make_drug_dict

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=int, default=0, help="model type: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE")
# parser.add_argument("-o", "--object", type=int, default=1, help="decoding object: 0:atoms, 1:bonds, 2:cell_line")
# parser.add_argument("-g", "--gpu", type=int, default=1, help="gpu number")
parser.add_argument("-b", "--branch", type=str, default='001', help="branch")
# parser.add_argument("-a", "--annotation", type=int, default=2, help="annotation type: 0:numbers, 1:heatmap, 2:both")

args = parser.parse_args()
model_type = args.model
# decoding_object = args.object
# gpu = args.gpu
b = args.branch
# annotation = args.annotation

model_name = ['GCN', 'GAT', 'GAT_Edge', 'GATv2', 'SAGE', 'GIN', 'GINE'][model_type]
branch_folder = "root_folder/root_" + b

# dataset = 'GDSC'
# test_data = TestbedDataset(root=branch_folder, dataset=dataset+'_test_mix')
# test_batch = 1
# test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

ss_path = branch_folder + '/Saliency/IG/CellLine/' + model_name
save_path = branch_folder + '/SaliencyMap/IG/CellLine/' + model_name

# if annotation == 0:
#     d_save_path = d_save_path + '/saliency_score'
# elif annotation == 1:
#     d_save_path = d_save_path + '/heatmap'
# elif annotation == 2:
#     d_save_path = d_save_path + '/ss+heatmap'
# else:
#     print('wrong annotation type!')
#     exit
        
os.makedirs(save_path, exist_ok=True)

_, _, genes = save_gene_expr_matrix_X()
genes = genes.values
gene_list = np.array([g.split(' (')[0] for g in genes])
drug_dict, sal_dict =  make_drug_dict(ss_path)
norm_sal_dict = normalize_ss(sal_dict)
rank_dict = rank_ss(norm_sal_dict)
draw_gene_saliency(rank_dict, norm_sal_dict, gene_list, save_path)