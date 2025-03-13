import pandas as pd
import re

data = "prism"
df = pd.read_csv(f"shap_values_LCO_{data}_cell-lines.csv")
names_df = pd.read_csv(f"../CCLE_expression_{data}19_w20.csv")

new_column_names = names_df.columns[1:].values  

new_column_names = [re.sub(r"\s*\(\d+\)", "", col) for col in new_column_names]

df.columns = [df.columns[0]] + list(new_column_names) + [df.columns[-1]]

df.to_csv(f"shap_values_LCO_{data}_cell-lines_with_gene_names.csv", index=False)
