import pandas as pd
import numpy as np
from pandas import DataFrame


filesNames = pd.read_csv('defectprediction.csv')  # Read all files names paths
path = filesNames.iloc[:, 1]  # select column 1, all rows
path = np.array(path)  # Convert to numpy array

for i in range(path.shape[0]):
    last = path[i].split('\\')
    print(last[-1])
    new = last[-1]
    df = DataFrame(columns=[new])
    df.to_csv('names.csv', mode='a')  # Write file names in csv
