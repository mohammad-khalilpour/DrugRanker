import pandas as pd
import numpy as np 
from pathlib import Path
import ast

def read_txt():
    pass

def save_results():
    pass

def main():
    # Define the log file path
    exp_dir = Path("/media/external_10TB/faraz_sarmeili/DrugRanker/expts/result")
    setup = "LCO"
    ds_name = "ctrp"
    target_model = "neuralndcg"
    representation = "layered_rdkit"
    results_dir = exp_dir / setup / ds_name / target_model / representation
    file_path = list(results_dir.rglob("./logs/avg*.txt"))[0]

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
    
    df = pd.DataFrame(logs_list)
    print(df)
    # columns = ["mode", "epoch", "fold"]
    # METRICS = ['CI', 'lCI', 'sCI', 'ktau', 'sp']
    # Kpos = [1,3,5,10,20,40,60]
    # for k in Kpos:
    #     METRICS += [f'AP@{k}', f'AH@{k}', f'NDCG@{k}']
    # columns += METRICS

    # results_df = pd.read_csv(file_path, header=None, names=columns)
    # print(results_df)

if __name__ == "__main__":
    main()