# XGDP

## Environment

```
conda env create --file=environment.yml
```

---

## Data Preparation

### Download the raw data

Download the drug response data in IC50 called PANCANCER_IC from [GDSC](https://www.cancerrxgene.org/downloads/drug_data). And download the gene expression data called CCLE_expression from [CCLE](https://depmap.org/portal/download/all/) under mRNA expression. 

### Preprocess the data

Place the PANCANCER_IC data under folder `data/GDSC` and place the CCLE_expression data under folder `data/CCLE`. Then run
```
python load_data.py <branch_num>
```

## Train the model

```
python train.py \
        --model <model_num>
        --branch <branch_num>
        --do_cv
        --do_attn
```
- Available models: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE, 7:WIRGAT, 8:ARGAT, 9:RGCN, 10:FiLM

## Explain the model

### Attribute the chemical structures with GNNExplainer

```
python gnnexplainer.py \
        --model <model_num>
        --branch <branch_num>
        --do_attn
        --explain_type <type>
python draw_gnnexplainer.py \
        --model <model_num>
        --branch <branch_num>
        --explain_type <type>
        --annotation <type>
```
- Available explaining types: 0:model, 1:phenomenon
- Available annotation types: 0:numbers, 1:heatmap, 2:both, 3:functional group-level heatmap

### Attribute the gene expression values with Integrated Gradients

```
python integrated_gradients.py \
        --model <model_num>
        --branch <branch_num>
        --do_attn
        --iqr_baseline
```
Refer to `pathway_analysis.ipynb` for the pathway analysis experiments based on the gene saliency scores.