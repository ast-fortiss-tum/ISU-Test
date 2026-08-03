import pickle
import numpy as np

with open('/Users/id/Downloads/agora/smplx_gt/trainset_3dpeople_adults_body/10011_w_Myriam_0_0.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data.keys())

exp = np.array(data.get("expression", []), dtype=float).tolist()
print(f"Expression shape: {np.array(exp)}")
