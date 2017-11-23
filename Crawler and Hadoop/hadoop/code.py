import pandas as pd

df = pd.read_excel("visual-p4.xls")

df_new = pd.DataFrame([df.ix[idx] for idx in df.index for _ in range(df.ix[idx]['# Conf'])]).reset_index(drop=True)

del df_new["# Conf"]

df_new.to_excel("visual-p4.xlsx")