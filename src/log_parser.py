from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def get_losses(log_content):
    probs_dict = {"loss": [],
                "gnorm": []}

    # Broader search for any lines containing the model names
    model_lines = []
    for i, l in enumerate(log_content):
        if "| INFO | Loss at epoch" in l:
            epoch = l.split('=')[-1].split(':')[0].strip(' ')
            loss_value = l.split('=')[-1].split(':')[-1].strip(' ')
            probs_dict["loss"].append((int(epoch), float(loss_value)))
        elif "| INFO | GNorm at epoch" in l:
            epoch = l.split('=')[-1].split(':')[0].strip(' ')
            gnorm_value = l.split('=')[-1].split(':')[-1].strip(' ')
            probs_dict["gnorm"].append((int(epoch), float(gnorm_value)))
        else:
            pass

    return probs_dict


def main():
    exp_dir = Path("/media/external_16TB_1/kian_khalilpour/DrugRanker/expts/result/")
    setup = "LCO"
    ds_name = "ctrp"
    target_model = "lambdaloss"
    representation = "atom_pair"
    results_dir = exp_dir / setup / ds_name / target_model / representation / "logs"

    # Define the log file path
    file_path = results_dir / 'train_0.log'

    # Open and read the file
    with open(file_path, 'r') as file:
        log_content = file.readlines()

    info_dict = get_losses(log_content)
    loss_epoch_list = info_dict["loss"]
    loss_epoch_dict = {k: v for k, v in loss_epoch_list}
    plt.plot(list(loss_epoch_dict.keys()), list(loss_epoch_dict.values()))
    save_dir = Path("/media/external_16TB_1/kian_khalilpour/DrugRanker/assets/loss")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{target_model}_{representation}_loss_0.png", dpi=300)
    plt.close()

    gnorm_epoch_list = info_dict["gnorm"]
    gnorm_epoch_dict = {k: v for k, v in gnorm_epoch_list}
    plt.plot(list(gnorm_epoch_dict.keys()), list(gnorm_epoch_dict.values()))
    save_dir = Path("/media/external_16TB_1/kian_khalilpour/DrugRanker/assets/gnorm")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{target_model}_{representation}_gnorm_0.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()