from __future__ import division, print_function
import numpy as np
from smote.SMOTE import Smote
import pandas as pd


features = pd.read_excel('ur_url')
features = features.drop('type', axis=1)
features = np.array(features)
smote = Smote(sampling_rate=100, k=5)
result = Smote().fit(features)

df = pd.DataFrame(data=result)
with pd.ExcelWriter('ur_url') as writer:
     df.to_excel(writer, sheet_name='1')