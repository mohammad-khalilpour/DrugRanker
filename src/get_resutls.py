import pandas as pd
import numpy as np 
from pathlib import Path


def read_txt():
    pass

def save_results():
    pass

def main():
    # Define the log file path
    exp_dir = Path("/media/external_10TB/faraz_sarmeili/DrugRanker/expts/result")
    setup = "LCO"
    ds_name = "ctrp"
    target_model = "listall"
    representation = "atom_pair"
    results_dir = exp_dir / setup / ds_name / target_model / representation
    
    print(results_dir)
    folds_test_paths = list(results_dir.rglob("./logs/test*.txt"))
    columns = ["num_epoch", "cell_line", ""]
    
    file_path = folds_test_paths[0]
    print(pd.read_csv(file_path))

if __name__ == "__main__":
    main()