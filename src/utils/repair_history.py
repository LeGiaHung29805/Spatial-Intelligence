import pandas as pd
df_test = pd.read_csv("data/processed/dynamic_landslide_fusion.csv")
print("Header của file là:", df_test.columns.tolist())