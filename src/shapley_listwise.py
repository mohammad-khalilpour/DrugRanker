import numpy as np
import torch
from scipy.stats import kendalltau
from dataloader.utils import *
from models.ranknet import RankNet
from utils.args import parse_args
from utils.common import *
from dataloader.loader import MoleculePoint, MoleculeDatasetTest, CellLine, MoleculeDatasetTrain
from functools import partial
import shap
from shap.utils._legacy import convert_to_model
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy
from models.loss import LambdaLoss


args = parse_args()

if args.data_path is None:
    args.data_path = "../data/ctrp/LRO/aucs.txt"

if args.smiles_path is None:
    args.smiles_path = "../data/ctrp/cmpd_smiles.txt"

args.genexp_path = "../data/CCLE/CCLE_expression.csv"

if args.device is None:
    args.device = "cpu"
args.feature_gen = "rdkit2d_atompair"


def rank_list(vector):
    temp = vector.argsort(descending=True)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(vector) + 1)

    return ranks


def features():
    data = get_data(args.data_path, args.smiles_path)
    features = precompute_features(args)
    clobj = CellLine(args.genexp_path)

    auc_points = []

    for d in data:
        auc_points.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=False))

    dataset = MoleculeDatasetTest(auc_points)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=100, collate_fn=dataset.collate_fn)

    features_batch = []
    clids = []
    for i, batch in enumerate(dataloader):
        for d in batch:
            features_batch.append(d.features)
            clids.append(d.clid)


    cl_emb = torch.from_numpy(np.array(clobj.get_expression(np.unique(clids, axis=0)))).to(args.device)
    background_cl_emb = cl_emb[1:3]
    test_cl_emb = cl_emb[3:5]
    original_cell_line = cl_emb[0]

    features = np.unique(features_batch, axis=0)

    background = numpy.array(background_cl_emb)
    test_data = numpy.array(test_cl_emb)

    model = RankNet(args).to(args.device)
    model.load_state_dict(torch.load("../saved_model/lambdarank_rdkit2.pt", weights_only=False, map_location=torch.device('cpu')))
    model.eval()

    mask = np.load("../s_g_prism.npy")
    mask = np.where(mask != 0, 1, 0)

    def model_predict(
            cell_line_batch,
            similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
            mixed_type_input=False,
    ):
        cell_line_batch = torch.tensor(cell_line_batch, dtype=torch.float32).to(args.device)
        batch_size = cell_line_batch.shape[0]
        full_cell_lines = torch.tile(background_cl_emb[0].unsqueeze(0), (batch_size, 1)).to(args.device)
        full_cell_lines = full_cell_lines.float()
        full_cell_lines[:, mask == 1] = cell_line_batch
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

    explainer = shap.KernelExplainer(
        model_predict,
        background[:, mask == 1]
    )

    shap_values = explainer.shap_values(test_data[:, mask == 1])

    # shap_exp = shap.Explanation(values=shap_values,
    #                             base_values=explainer.expected_value,
    #                             data=test_cl_emb[:, mask == 1])

    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"Feature_{i}" for i in range(shap_values.shape[1])]
    )

    shap_df["Base Value"] = explainer.expected_value

    shap_df.to_csv("shap_values.csv", index=False)

    print(shap_df.head())

with torch.no_grad():
    features()

