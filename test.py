import os

print("Current working directory:", os.getcwd())
print("xlsx exists?", os.path.exists("df_selected.xlsx"))
print("json exists?", os.path.exists("arrival_model_params.json"))