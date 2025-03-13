import numpy as np
import torch
from scipy.stats import kendalltau
from dataloader.utils import *
from models.ranknet import RankNet
from utils.args import parse_args
from utils.common import *
from dataloader.loader import MoleculePoint, MoleculeDatasetTest, CellLine
from functools import partial
import shap
from shap.utils._legacy import convert_to_model
from torch.utils.data import DataLoader
import pandas as pd


args = parse_args()

data_set = "prism"
cancers_types_file = f"../data/{data_set}/LRO/cancer.txt"
cell_lines_file = f"../data/{data_set}/LRO/cells.txt"

with open(cancers_types_file, 'r') as file:
    cancer_list = [line.strip() for line in file]

with open(cell_lines_file, 'r') as file:
    cell_line_list = [line.strip() for line in file]

cancer_cell_lines = [cell_line_list[index] for index, cancer in enumerate(cancer_list) if cancer == "Breast Cancer"]

args.data_path = f"../data/{data_set}/LRO/aucs.txt"
args.smiles_path = f"../data/{data_set}/cmpd_smiles.txt"

args.genexp_path = "../data/CCLE/CCLE_expression.csv"

if args.device is None:
    args.device = "cpu"
args.feature_gen = "rdkit2d_morganc"
args.background_samples = 2


def rank_list(vector):
    temp = vector.argsort(descending=True)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(vector) + 1)

    return ranks


def cell_lines():
    data = get_data(args.data_path, args.smiles_path)
    features = precompute_features(args)
    clobj = CellLine(args.genexp_path)

    auc_points = []

    for d in data:
        auc_points.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=False))

    dataset = MoleculeDatasetTest(auc_points)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=100, collate_fn=dataset.collate_fn)

    features_set = set()
    clids_set = set()

    for i, batch in enumerate(dataloader):
        print(i)
        for d in batch:
            features_set.add(tuple(d.features))
            clids_set.add(d.clid)

    print("done")

    clids = np.array(list(clids_set))
    features = np.array(list(features_set), dtype=np.float64)
    print(features.shape)

    cl_emb = torch.from_numpy(np.array(clobj.get_expression(clids))).to(args.device)

    cancer_indices = np.where(np.isin(clids, cancer_cell_lines))[0]
    test_cl_emb = cl_emb[cancer_indices].numpy()


    model = RankNet(args).to(args.device)
    model.load_state_dict(torch.load("../saved_model/LCO-prism-lambdarank.pt", weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    indices = np.load("../selected_genes_indices_prism19.npy")

    def model_predict(
            cell_line_batch,
            original_cell_line,
            similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    ):
        cell_line_batch = torch.tensor(cell_line_batch, dtype=torch.float32).to(args.device)
        batch_size = cell_line_batch.shape[0]
        full_cell_lines = torch.tile(original_cell_line.unsqueeze(0), (batch_size, 1)).to(args.device)
        full_cell_lines = full_cell_lines.float()
        full_cell_lines[:, indices] = cell_line_batch
        pred = model(
            clines=original_cell_line,
            feat1=features,
            output_type=0
        )
        og_rank = rank_list(pred)

        scores = []
        for cell_line in full_cell_lines:
            new_pred = model(
                clines=cell_line,
                feat1=features,
                output_type=0
            )
            new_rank = rank_list(new_pred)
            scores.append(similarity_coefficient(og_rank, new_rank))

        return np.array(scores)


    output_file = "shap_values_LCO_prism_cell-lines.csv"

    columns = ["Cell Line"] + [f"Feature_{i}" for i in indices] + ["Base Value"]

    if not os.path.exists(output_file):
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    for i, cell_line in enumerate(test_cl_emb):
        print(f"Processing {i}/{len(test_cl_emb)}")
        
        explainer = shap.KernelExplainer(
            convert_to_model(
                partial(model_predict, original_cell_line=torch.from_numpy(cell_line))
            ),
            shap.sample(np.array(cl_emb), 4)[:, indices]
        )
        
        shap_values = explainer.shap_values(cell_line[indices])
        expected_value = explainer.expected_value

        row_df = pd.DataFrame([[clids[cancer_indices[i]]] + np.append(shap_values, expected_value).tolist()], columns=columns)

        row_df.to_csv(output_file, mode="a", header=False, index=False)


def features():
    data = get_data(args.data_path, args.smiles_path)
    features = precompute_features(args)
    clobj = CellLine(args.genexp_path)

    auc_points = []

    for d in data:
        auc_points.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=False))

    dataset = MoleculeDatasetTest(auc_points)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=100, collate_fn=dataset.collate_fn)

    features_set = set()
    clids_set = set()

    for i, batch in enumerate(dataloader):
        print(i)
        for d in batch:
            features_set.add(tuple(d.features))
            clids_set.add(d.clid)

    print("done")

    clids = np.array(list(clids_set))
    features = np.array(list(features_set), dtype=np.float64)
    print(features.shape)

    cl_emb = torch.from_numpy(np.array(clobj.get_expression(clids))).to(args.device)

    cancer_indices = np.where(np.isin(clids, cancer_cell_lines))[0]
    test_cl_emb = cl_emb[cancer_indices].numpy()
    

    model = RankNet(args).to(args.device)
    model.load_state_dict(torch.load("../saved_model/LCO-prism-lambdarank.pt", weights_only=True, map_location=torch.device('cpu')))
    
    model.eval()
    print("another another done")
    def model_predict(
            features_array,
            features,
            cell_line,
            similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    ):

        pred = model(
            clines=cell_line,
            feat1=features,
            output_type=0
        )
        og_rank = rank_list(pred)

        scores = []

        for a in features_array:
            adjusted_feature = np.array([np.where(pd.isna(a), q, a) for q in features])
            new_pred = model(
                clines=cell_line,
                feat1=adjusted_feature,
                output_type=0
            )
            new_rank = rank_list(new_pred)

            scores.append(similarity_coefficient(og_rank, new_rank))

        # adjusted_features = np.array(
        #     [[np.where(pd.isna(a), q, a) for q in features] for a in features_array]
        # )

        # scores = []
        # for features in adjusted_features:
            
        #     new_pred = model(
        #         clines=cell_line,
        #         feat1=features,
        #         output_type=0
        #     )
        #     new_rank = rank_list(new_pred)

        #     scores.append(similarity_coefficient(og_rank, new_rank))
        return np.array(scores)

    output_file = "shap_values_LCO_prism_drugs.csv"

    columns = ["Cell Line"] + [f"Feature_{i}" for i in range(features[0].shape[0])] + ["Base Value"]

    if not os.path.exists(output_file):
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    for i, cell_line in enumerate(test_cl_emb):
        print(f"Processing {i}/{len(test_cl_emb)}")
        
        explainer = shap.KernelExplainer(
            convert_to_model(
                    partial(model_predict, features=features, cell_line=torch.from_numpy(cell_line))
                ),
                shap.sample(features, args.background_samples)
        )
        
        shap_values = explainer.shap_values(np.array([np.full(features[0].shape, None)], dtype=np.float64))
        expected_value = explainer.expected_value

        row_df = pd.DataFrame([[clids[cancer_indices[i]]] + np.append(shap_values, expected_value).tolist()], columns=columns)

        row_df.to_csv(output_file, mode="a", header=False, index=False)

with torch.no_grad():
    # features()
    cell_lines()




