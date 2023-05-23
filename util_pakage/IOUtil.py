import pandas as pd;
import numpy as np

class CSVDataLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_csv(self):
        df = pd.read_csv(self.filename)
        data_array =np.array(df[['longitude','latitude']], dtype=np.float32)
        for i in range(len(df)):
            data_array[i] = np.array([df['longitude'][i], df['latitude'][i]],dtype=np.float32)

        return data_array

