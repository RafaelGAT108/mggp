import pandas as pd
from mggp import MGGP

df_train = pd.read_csv("F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level3.csv").to_numpy()
# u_train, y_train = (df_train[:7372, :2], df_train[:7372, 2:5])
u_train, y_train = (df_train[:, :2], df_train[:, 2:5])
df_val = pd.read_csv("F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level2_Validation.csv").to_numpy()
u_val, y_val = (df_val[:, :2], df_val[:, 2:5])
print(len(u_train), len(y_train))

mggp = MGGP(inputs=u_train,
            outputs=y_train,
            validation=(u_val, y_val),
            nDelays=[1, 2, 3, 5, 10, 15, 25, 50],
            # nDelays=2,
            generations=50,
            evaluationMode="RMSE",
            evaluationType="FreeRun")

mggp.run()
