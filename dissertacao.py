import numpy as np
import pandas as pd
import os
from mggp import MGGP

if __name__ == '__main__':
        
    df_train = pd.read_csv("angle_slip.csv").to_numpy()
    u, y = (df_train[:, :2], df_train[:, 2:3])

    split = 0.8
    y_train = y[:int(len(y)*split)]
    y_val = y[int(len(y)*split):]

    print(len(y_val), len(y_train))

    u_train = u[:int(len(u)*split)]
    u_val = u[int(len(u)*split):]

    print(len(u_val), len(u_train))

    mggp = MGGP(inputs=u_train,
                outputs=y_train,
                validation=(u_val, y_val),
                #nDelays=[1, 2, 3, 5, 10, 15, 25, 50],
                nDelays=15,
                generations=1,
                evaluationMode="RMSE",
                k=10,
                evaluationType="MShooting")

    mggp.run()