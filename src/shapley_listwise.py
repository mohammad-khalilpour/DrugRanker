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
import pandas as pd
from pathlib import Path


args = parse_args()


def rank_list(vector):
    temp = vector.argsort(descending=True)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(vector) + 1)

    return ranks


with torch.no_grad():

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
    test_cl_emb = cl_emb[:args.test_numbers].numpy()

    model = RankNet(args).to(args.device)
    fold_path = Path(args.save_path) / f'fold_{args.fold_num}'
    model_path = list(fold_path.rglob("./epoch_1*.pt"))[0]
    model.load_state_dict(
        torch.load(model_path, weights_only=False, map_location=torch.device(args.device)))
    model.eval()


    if args.described_features == "cell_line":
        background_cl_emb = cl_emb[args.test_numbers:args.test_numbers + args.background_samples]

        features = np.unique(features_batch, axis=0)

        background = numpy.array(background_cl_emb)
        test_data = numpy.array(test_cl_emb)

        indices = np.load("../selected_genes_indices_ctrp.npy")

        def model_predict(
                cell_line_batch,
                original_cell_line,
                similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
                mixed_type_input=False,
        ):
            cell_line_batch = torch.tensor(cell_line_batch, dtype=torch.float32).to(args.device)
            batch_size = cell_line_batch.shape[0]
            full_cell_lines = torch.tile(background_cl_emb[0].unsqueeze(0), (batch_size, 1)).to(args.device)
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


        shap_values_list = []
        expected_values_list = []

        for i, cell_line in enumerate(test_cl_emb):
            print(i)
            explainer = shap.KernelExplainer(
                convert_to_model(
                    partial(model_predict, original_cell_line=torch.from_numpy(cell_line))
                ),
                background[:, indices]
            )
            shap_values = explainer.shap_values(cell_line[indices])

            shap_values_list.append(shap_values)
            expected_values_list.append(explainer.expected_value)

        shap_df = pd.DataFrame(
            np.array(shap_values_list),
            columns=[f"Feature_{i}" for i in indices]
        )

        shap_df["Base Value"] = expected_values_list

        shap_df.to_csv(f"{fold_path}_cell_line.csv", index=False)

    elif args.described_features == "drug":

        features = np.unique(features_batch, axis=0).astype(np.float64)
        test_data = numpy.array(test_cl_emb)

        def model_predict(
                features_array,
                features,
                cell_line,
                similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
                mixed_type_input=False,
        ):

            pred = model(
                clines=cell_line,
                feat1=features,
                output_type=0
            )
            og_rank = rank_list(pred)

            adjusted_features = np.array(
                [[np.where(pd.isna(a), q, a) for q in features] for a in features_array]
            )

            scores = []
            for features in adjusted_features:
                new_pred = model(
                    clines=cell_line,
                    feat1=features,
                    output_type=0
                )
                new_rank = rank_list(new_pred)

                scores.append(similarity_coefficient(og_rank, new_rank))
            return np.array(scores)

        shap_values_list = []
        expected_values_list = []

        for i, cell_line in enumerate(test_cl_emb):
            print(i)
            explainer = shap.KernelExplainer(
                convert_to_model(
                    partial(model_predict, features=features, cell_line=torch.from_numpy(cell_line))
                ),
                shap.sample(features[args.test_number:], args.background_samples)
            )
            shap_values = explainer.shap_values(np.array([np.full(features[0].shape, None)], dtype=np.float64))

            shap_values_list.append(shap_values)
            expected_values_list.append(explainer.expected_value)

        shap_df = pd.DataFrame(
            np.array(shap_values_list),
            columns=[f"Feature_{i}" for i in range(features[0].shape)]
        )

        shap_df["Base Value"] = expected_values_list

        shap_df.to_csv(f"{fold_path}shap_values_drug.csv", index=False)


