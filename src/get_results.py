import pandas as pd
import numpy as np 
from pathlib import Path
import ast
from utils.args import parse_args


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

def calc_results(args, file_paths):
    print(file_paths)
    dfs_list = [get_df_from_txt(p) for p in file_paths]
    max_num_epoch = min([df["epoch"].max() for df in dfs_list])
    modes = np.unique([np.unique(df["mode"].values) for df in dfs_list])

    if "Inference" in modes:
        log_steps = max_num_epoch
    else:
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
    return avg_results_df, max_num_epoch

def get_avg_attention(args):
    cell_attn_avg_df = None
    drug_attn_avg_df = None
    if any(cname in args.update_emb for cname in ["ppi", "cell"]):
        cell_attn_fpaths = list(Path(args.save_path).rglob("fold_*/cell_aw*.csv"))
        cell_concat_df = pd.concat([pd.read_csv(fpath, index_col=0) for fpath in cell_attn_fpaths])
        cell_attn_avg_df = cell_concat_df.groupby(cell_concat_df.index).mean()
    if "drug" in args.update_emb:
        drug_attn_fpaths = list(Path(args.save_path).rglob("fold_*/frug_aw*.csv"))
        drug_concat_df = pd.concat([pd.read_csv(fpath, index_col=0) for fpath in drug_attn_fpaths])
        drug_attn_avg_df = drug_concat_df.groupby(drug_concat_df.index).mean()

    return cell_attn_avg_df, drug_attn_avg_df

def main(args):
    # Define the log file path
    
    results_dir = Path(args.save_path)
    file_paths = list(results_dir.rglob("logs/results*.txt"))
    
    if (args.do_explain) & ("attention" in args.update_emb):
        cell_df, drug_df = get_avg_attention(args)
        print(cell_df)
        if cell_df is not None:
            cell_df.to_csv((results_dir / "logs/cell_attn_avg.csv"))
        if drug_df is not None:
            drug_df.to_csv((results_dir / "logs/drug_attn_avg.csv"))

    if args.do_test: 
        file_paths = [f for f in file_paths if "inference" in f.name]
    else:
        file_paths = [f for f in file_paths if "inference" not in f.name]

    avg_results_df, max_num_epoch = calc_results(args, file_paths)
    if args.do_test:
        avg_results_df.to_csv((results_dir / "logs/results_inference_avg.csv"), index=False)
    else:
        avg_results_df.to_csv((results_dir / "logs/results_avg.csv"), index=False)
    
    if args.do_test:
        target_modes = ["Inference"]
    else:
        target_modes = ["TEST"]
    for m in target_modes:
        print(f"Results for {m} Set")
        for c in avg_results_df.columns:
            print("    ", end="")
            print(f"{c}: {avg_results_df[(avg_results_df['epoch']==max_num_epoch) & (avg_results_df['mode']==m)][c].values[0]}")

if __name__ == "__main__":
    main(parse_args())
