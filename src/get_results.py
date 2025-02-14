import pandas as pd
import numpy as np 
from pathlib import Path
import torch
import ast
from utils.args import parse_args
from utils.common import load_model
from models.ranknet import RankNet

def get_df_from_txt(file_path):
    # Open and read the file
    with open(file_path, 'r') as file:
        log_content = file.readlines()

    # Broader search for any lines containing the model names
    logs_list = []
    for i, l in enumerate(log_content):
        if l[0] == "{":
            logs_list.append(ast.literal_eval(l))

    ## check the length of the dict
    it = iter(logs_list)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('not all lists have same length!')
    
    return pd.DataFrame(logs_list)

def get_model(model_path, args):
    model = RankNet(args)
    # load_model(model, model_path, device=args.device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    return model

def inspect_saved_models(args):
    '''exp_dir = Path("/media/external_10TB/faraz_sarmeili/DrugRanker/expts/result")
    setup = "LCO"
    ds_name = "ctrp"
    target_model = "lambdarank"
    representation = "atom_pair"
    results_dir = exp_dir / setup / ds_name / target_model / representation'''
    args.device = torch.device(args.desired_device if torch.cuda.is_available() and args.cuda else "cpu")
    results_dir = Path(args.save_path)
    file_paths = list(results_dir.rglob("logs/results*.txt"))

    model_path = args.save_path + f"fold_{args.only_fold+1}/epoch_1.pt"
    print(f"model_path: {model_path}")
    model = get_model(model_path, args)
    print("model.state_dict()")
    print(model.enc)

def main(args):
    # Define the log file path
    '''exp_dir = Path("/media/external_10TB/faraz_sarmeili/DrugRanker/expts/result")
    setup = "LCO"
    ds_name = "ctrp"
    target_model = "lambdarank"
    representation = "atom_pair"
    results_dir = exp_dir / setup / ds_name / target_model / representation'''
    results_dir = Path(args.save_path)
    file_paths = list(results_dir.rglob("logs/results*.txt"))

    dfs_list = [get_df_from_txt(p) for p in file_paths]
    max_num_epoch = min([df["epoch"].max() for df in dfs_list])
    modes = np.unique([np.unique(df["mode"].values) for df in dfs_list])
    log_steps = args.log_steps

    # concatenate them
    df_concat = pd.concat(dfs_list)
    avg_results = []
    for i in range(log_steps, max_num_epoch+1, log_steps):
        for m in modes:
            row = df_concat[(df_concat["epoch"]==i) &
                            (df_concat["mode"]==m)].iloc[:, :-2].mean()
            row['mode'] = m
            row['epoch'] = i
            avg_results.append(row)

    avg_results_df = pd.concat(avg_results, axis=1).T
    avg_results_df.to_csv((results_dir / "logs/results_avg.csv"), index=False)
    
    target_modes = ["TEST"]
    for m in target_modes:
        print(f"Results for {m} Set")
        for c in avg_results_df.columns:
            print("    ", end="")
            print(f"{c}: {avg_results_df[(avg_results_df['epoch']==max_num_epoch) & (avg_results_df['mode']==m)][c].values[0]}")

if __name__ == "__main__":
    args = parse_args()

    if args.get_results_gnn == "aggregate-results":
        main(args)
    elif args.get_results_gnn == "model-inspection":
        inspect_saved_models(parse_args())
    else: 
        pass