import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import os

def make_drug_dict(dir):
    drug_dict = dict()
    sal_dict = dict()
    for filename in os.listdir(dir):
        drug_name = filename.split('_')[1]
        one = np.load(os.path.join(dir, filename))
        if drug_name not in drug_dict.keys() and drug_name not in sal_dict.keys():
            drug_dict[drug_name] = 1
            sal_dict[drug_name] = one
        else:
            drug_dict[drug_name] += 1
            sal_dict[drug_name] = np.add(sal_dict[drug_name], one)
    
    for k, v in sal_dict.items():
        sal_dict[k] = v/drug_dict[k]
    
    return drug_dict, sal_dict


def make_edge_dict(loader):
    smiles_dict = dict()
    edge_index_dict = dict()
    for data in loader:
        drug_name = data.drug_name[0]
        smiles = data.smiles[0]
        edge_index = data.edge_index.numpy()
        
        if drug_name not in smiles_dict.keys() and drug_name not in edge_index_dict.keys():
            smiles_dict[drug_name] = smiles
            edge_index_dict[drug_name] = edge_index
    
    return smiles_dict, edge_index_dict


def draw_mol_saliency_scores(drug_sal_dict, smiles_dict, edge_index_dict, save_path, annotation_type):
    for k, v in drug_sal_dict.items():
        print('working on ', k)
        edge_index = edge_index_dict[k]
        # norm_v = (v - v.min()) /(v.max() - v.min())

        edge_ss_dict = defaultdict(float)
        counts = defaultdict(int)

        # for val, x, y in zip(norm_v, *edge_index):
        for val, x, y in zip(v, *edge_index):
            if x > y:
                x, y = y, x
            edge_ss_dict[(x, y)] += val
            counts[(x, y)] += 1
            
        for edge, count in counts.items():
            edge_ss_dict[edge] /= count
            # edge_ss_dict[edge] = edge_ss_dict[edge].round(2)
            
        min_ss = min(edge_ss_dict.values())
        max_ss = max(edge_ss_dict.values())
        # print(min_ss, max_ss)

        for edge, value in edge_ss_dict.items():
            edge_ss_dict[edge] = (value - min_ss)/(max_ss - min_ss)
            edge_ss_dict[edge] = edge_ss_dict[edge].round(2)

        smiles = smiles_dict[k]
        mol = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol)
        AllChem.Compute2DCoords(mol)
        bond_weights = []
        for i, bond in enumerate(mol.GetBonds()):
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u > v:
                u, v = v, u
            if annotation_type == 0:
                bond.SetProp('bondNote',str(edge_ss_dict[(u, v)]))
            elif annotation_type == 1:
                bond_weights.append(edge_ss_dict[(u, v)])
            elif annotation_type == 2:
                bond.SetProp('bondNote',str(edge_ss_dict[(u, v)]))
                bond_weights.append(edge_ss_dict[(u, v)])
        
        if annotation_type == 0:
            Chem.Draw.MolToImageFile(mol, os.path.join(save_path, k + '.png'), size = (1000, 1000))
        elif annotation_type == 1 or annotation_type == 2:
            canvas = mapvalues2mol(mol, bond_weights = bond_weights)
            img = transform2png(canvas.GetDrawingText())
            img.save(os.path.join(save_path, k + '.png'))


def normalize_ss(sal_dict):
    for k, v in sal_dict.items():
        max_ss = np.max(v)
        min_ss = np.min(v)
        sal_dict[k] = (v - min_ss) / (max_ss - min_ss)
    
    return sal_dict


def rank_ss(sal_dict):
    rank_dict = dict()
    for k, v in sal_dict.items():
        temp = np.argsort(-1*(v.reshape(-1)))
        rank_dict[k] = temp
        
    return rank_dict


def draw_one_drug(save_path, drug_name, ranked_ss, ranked_genes, top_n=25):
    values = ranked_ss[:top_n]
    index = ranked_genes[:top_n]
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(250, 20))
    sns.set(font_scale=5)


    res = sns.heatmap(values.reshape(1, -1))
    # plt.xticks(np.arange(100), temp[:100])
    res.set_xticklabels(index)
    res.set_yticklabels([])

    # plt.show()
    fig.savefig(os.path.join(save_path, drug_name + '.png'))


def draw_gene_saliency(rank_dict, sal_dict, gene_list, save_path):
    i = 0
    for drug in sal_dict.keys():
        i += 1
        print('working on ', drug)
        print('progress: ', i, '/', len(sal_dict))
        rnk = rank_dict[drug]
        sal_score = sal_dict[drug].reshape(-1)
        ranked_ss = sal_score[rnk]
        ranked_genes = gene_list[rnk]

        draw_one_drug(save_path, drug, ranked_ss, ranked_genes)