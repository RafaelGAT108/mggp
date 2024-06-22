import pandas as pd
from mggp import MGGP

df = pd.read_csv("F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level3.csv").to_numpy()
# df = pd.read_csv("F16GVT_Files/BenchmarkData/F16Data_SineSw_Level3.csv").to_numpy()
u, y = (df[:, :2], df[:, 2:5])

mggp = MGGP(inputs=u, outputs=y, generations=50)

mggp.run()
