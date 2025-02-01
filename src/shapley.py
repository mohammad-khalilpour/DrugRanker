import numpy as np
import torch

from dataloader.utils import *
from models.ranknet import RankNet
from utils.args import parse_args
from utils.common import *
from dataloader.loader import MoleculePoint, MoleculeDatasetTest, CellLine, MoleculeDatasetTrain
import shap
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy
from models.loss import LambdaLoss


args = parse_args()

if args.data_path is None:
    args.data_path = "../data/ctrp/LRO/aucs.txt"

if args.smiles_path is None:
    args.smiles_path = "../data/ctrp/cmpd_smiles.txt"

# if args.genexp_path is None:
args.genexp_path = "../data/CCLE/CCLE_expression.csv"

if args.device is None:
    args.device = "cpu"
args.feature_gen = "atom_pair"
# args.pretrained_ae = True
# args.trained_ae_path = "../saved_model/model.pt"


# args.pretrained_ae = True
def features():
    data = get_data(args.data_path, args.smiles_path)
    features = precompute_features(args)
    clobj = CellLine(args.genexp_path)

    auc_points = []

    for d in data:
        auc_points.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=False))

    dataset = MoleculeDatasetTest(auc_points)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=500, collate_fn=dataset.collate_fn)

    batch = next(iter(dataloader))

    features_batch = [d.features for d in batch]
    clids = [d.clid for d in batch]

    cl_emb = torch.from_numpy(np.array(clobj.get_expression(clids))).to(args.device)

    background_clids = cl_emb

    background_features = features_batch[:10]
    test_features = features_batch[-3:]

    background = numpy.array(background_features)
    test_data = numpy.array(test_features)

    model = RankNet(args).to(args.device)
    model.load_state_dict(torch.load("../saved_model/epoch_100_1.pt", weights_only=False, map_location=torch.device('cpu')))
    model.eval()

    # print(model(
    #     clines=cl_emb[:5],
    #     feat1=background_features[:5],
    #     output_type=0,
    # ))

    def model_predict(features):
        features = torch.tensor(features, dtype=torch.float32).to(args.device)
        batch_size = features.shape[0]
        clids_batch = background_clids.repeat((batch_size // background_clids.shape[0]) + 1, 1)
        clids_batch = clids_batch[:batch_size]

        with torch.no_grad():
            output = model(
                clines=clids_batch,
                feat1=features,
                output_type=0
            )

        return output.cpu().numpy()


    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(test_data)
    test_features = np.array(test_features)
    print(test_features.shape)

    shap_exp = shap.Explanation(values=shap_values,
                                base_values=explainer.expected_value,
                                data=test_features)

    shap.plots.waterfall(shap_exp[0], max_display=14)
    shap.plots.waterfall(shap_exp[1], max_display=14)
    shap.plots.waterfall(shap_exp[2], max_display=14)


def cell_lines():
    data = get_data(args.data_path, args.smiles_path)
    features = precompute_features(args)
    clobj = CellLine(args.genexp_path)

    auc_points = []

    for d in data:
        auc_points.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=False))

    dataset = MoleculeDatasetTest(auc_points)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=300, collate_fn=dataset.collate_fn)

    batch = next(iter(dataloader))

    features_batch = [d.features for d in batch]
    clids = [d.clid for d in batch]

    cl_emb = torch.from_numpy(np.unique(np.array(clobj.get_expression(clids)), axis=0)).to(args.device)
    print(cl_emb.shape)

    background_cl_emb = cl_emb[:2]
    test_cl_emb = cl_emb[3:6]

    background_features = torch.from_numpy(np.array(features_batch[:10]))

    background = numpy.array(background_cl_emb)
    test_data = numpy.array(test_cl_emb)

    model = RankNet(args).to(args.device)
    model.load_state_dict(torch.load("../saved_model/epoch_100_1.pt", weights_only=False, map_location=torch.device('cpu')))
    model.eval()


    def model_predict(cell_lines):
        cell_lines = torch.tensor(cell_lines, dtype=torch.float32).to(args.device)
        batch_size = cell_lines.shape[0]
        features_batch = background_features.repeat((batch_size // background_features.shape[0]) + 1, 1)
        features_batch = features_batch[:batch_size]

        with torch.no_grad():
            output = model(
                clines=cell_lines,
                feat1=features_batch,
                output_type=0
            )

        return output.cpu().numpy()

    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(test_data)

    shap_exp = shap.Explanation(values=shap_values,
                                base_values=explainer.expected_value,
                                data=test_cl_emb)

    shap.plots.waterfall(shap_exp[0], max_display=20)
    shap.plots.waterfall(shap_exp[1], max_display=20)
    shap.plots.waterfall(shap_exp[2], max_display=20)


if __name__ == "__main__":
    cell_lines()
