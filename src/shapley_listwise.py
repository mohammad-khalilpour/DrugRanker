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
args.feature_gen = "atom_pair"


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
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1000, collate_fn=dataset.collate_fn)

    batch = next(iter(dataloader))

    features_batch = [d.features for d in batch]
    clids = [d.clid for d in batch]

    cl_emb = torch.from_numpy(np.array(clobj.get_expression(clids))).to(args.device)

    background_clids = cl_emb[0]

    background_features = np.unique(features_batch, axis=0)[:10]
    test_features = np.unique(features_batch, axis=0)[-25:]

    background = numpy.array(background_features)
    test_data = numpy.array(test_features)

    model = RankNet(args).to(args.device)
    model.load_state_dict(torch.load("../saved_model/epoch_100_1.pt", weights_only=False, map_location=torch.device('cpu')))
    model.eval()

    def model_predict(
            array,
            features,
            similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
            mixed_type_input=False,
    ):
        features = torch.tensor(features, dtype=torch.float32).to(args.device)
        batch_size = features.shape[0]
        clids_batch = background_clids.repeat((batch_size // background_clids.shape[0]) + 1, 1)
        clids_batch = clids_batch[:batch_size]

        pred = model(
            clines=clids_batch,
            feat1=features,
            output_type=0
        )
        og_rank = rank_list(pred)

        if not mixed_type_input:
            adjusted_features = np.array(
                [[np.where(a == 0, q, a) for q in features] for a in array]
            )
        else:
            adjusted_features = np.array(
                [
                    [pd.Series(q).where(pd.isna(a), a).values for q in features]
                    for a in array
                ]
            )

        scores = []
        for features_background_sample in adjusted_features:
            new_pred = model(
                clines=clids_batch,
                feat1=features_background_sample,
                output_type=0
            )
            new_rank = rank_list(new_pred)

            scores.append(similarity_coefficient(og_rank, new_rank))
        print(scores)
        return np.array(scores)

    explainer = shap.KernelExplainer(
        convert_to_model(
            partial(model_predict, features=test_data)
        ),
        background
    )

    vector_of_zeros = np.array([np.full(test_data[0].shape, 0)])

    shap_values = explainer.shap_values(vector_of_zeros)
    test_features = np.array(test_features)

    shap_exp = shap.Explanation(values=shap_values,
                                base_values=explainer.expected_value,
                                data=test_features)

    shap.plots.waterfall(shap_exp[0], max_display=14)
    # shap.plots.waterfall(shap_exp[1], max_display=14)
    # shap.plots.waterfall(shap_exp[2], max_display=14)

with torch.no_grad():
    features()

