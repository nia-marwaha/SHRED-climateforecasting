import numpy as np
#from processdata import TimeSeriesDataset
import models
import torch
import os
import os.path
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataset(torch.utils.data.Dataset):
    '''Takes input sequence of sensor measurements with shape (batch size, lags, num_sensors)
    and corresponding measurments of high-dimensional state, return Torch dataset'''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len


folder_path = '/home/nia/Desktop/extracted_chemicals/svd'
chemicals = ['O3', 'NO', 'ISOP', 'NO2', 'OH']
for chemical in chemicals:
    U_file = os.path.join(folder_path, f'{chemical}_U.npy')
    S_file = os.path.join(folder_path, f'{chemical}_S.npy')
    V_file = os.path.join(folder_path, f'{chemical}_V.npy')
    if os.path.exists(U_file) and os.path.exists(S_file) and os.path.exists(V_file):
        globals()[f'{chemical}_u'] = np.load(U_file)
        globals()[f'{chemical}_s'] = np.load(S_file)
        globals()[f'{chemical}_v'] = np.load(V_file)
        print(f"Loaded SVD components for {chemical}:")
        print("-" * 40)
    else:
        print(f"Missing SVD components for {chemical}.")


u_total = None
s_total = None
v_total = None

for chemical in chemicals:
    u = globals()[f'{chemical}_u']
    s = globals()[f'{chemical}_s']
    v = globals()[f'{chemical}_v']
    
    if u_total is None:  
        u_total = u
        s_total = s
        v_total = v
    else: 
        u_total = np.hstack((u_total, u))
        s_total = np.vstack((s_total, s))
        v_total = np.vstack((v_total, v))

print("Final stacked components:")
print(f"u_total shape: {u_total.shape}")
print(f"s_total shape: {s_total.shape}")
print(f"v_total shape: {v_total.shape}")

first_chemical = chemicals[0]
path = '/home/nia/Desktop/extracted_chemicals'

X = np.load(os.path.join(path, f'{first_chemical}.npy'))
print(f"Loaded {first_chemical}")
X = X[:,:,10:-10, :]
print(f"size of chemical data:{X.shape}")
X = X.reshape(2016, -1, order = 'F').T
X_ = (X - np.mean(X, axis=0))
print('X_',X_.shape)

n1=X_.shape[0]
n2=X_.shape[1]
m2 = s_total.shape[1]
print(n1)
print(n2)
print(m2)

num_sensors = 10 
lags = 52

load_X = v_total.T

sensor_locations_ne = np.random.choice(n1, size=num_sensors, replace=False)
sensor_locations = list(range(num_sensors))
load_X = np.hstack((X_[sensor_locations_ne, :].T, load_X))
n = (load_X).shape[0]
m = (load_X).shape[1]

train_indices = np.random.choice(n - lags, size=500, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]

print('n',n)
print('valid_test',len(valid_test_indices))
print('train',len(train_indices))
print('valid',len(valid_indices))
print('test',len(test_indices))

sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)


all_data_in = np.zeros((n - lags, lags, num_sensors))
print(all_data_in.shape)
for j in range(len(all_data_in)):
    all_data_in[j] = transformed_X[j:j+lags, sensor_locations]


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)


train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

loss_list=[]
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=100, l2=1000, dropout=0.05).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=16, num_epochs=1000, lr=1e-4, verbose=True, patience=5)

test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
loss = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
loss_list.append(loss) 

print("all_data_in",all_data_in.shape)
print("train_data_in",train_data_in.shape)
print("test_data",len(test_dataset))
print("test_data_in",test_data_in.shape)
print("test_indices",test_indices.shape)
print("valid_test_indices",valid_test_indices.shape)
print("test_recons",test_recons.shape)

loss = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
print(loss)

base_dir = '/home/nia/Desktop/extracted_chemicals/'
chemical_dir = os.path.join(base_dir, first_chemical)
os.makedirs(chemical_dir, exist_ok=True)
np.save(os.path.join(chemical_dir, 'sensor_locations_ne.npy'), sensor_locations_ne)
np.save(os.path.join(chemical_dir, 'test_recons.npy'), test_recons)
np.save(os.path.join(chemical_dir, 'test_ground_truth.npy'), test_ground_truth)
np.save(os.path.join(chemical_dir, 'u_total.npy'), u_total)
np.save(os.path.join(chemical_dir, 's_total.npy'), s_total)
print(f"All arrays saved in directory: {chemical_dir}")