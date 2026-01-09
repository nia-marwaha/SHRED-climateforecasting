import numpy as np
import os
import os.path
from scipy.sparse.linalg import svds



def twohundred_mode_SVD(folder_path):
    svd_folder = os.path.join(folder_path, "svd")
    os.makedirs(svd_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            chemical_name = os.path.splitext(file_name)[0]  
            file_path = os.path.join(folder_path, file_name)
            matrix = np.load(file_path)
            matrix = matrix[:, :, 10:-10, :]
            print(f"Working on {chemical_name}:{matrix.shape}")
            reshaped_matrix = matrix.reshape(2016, -1, order = 'F').T
            print(f"size of chemical data:{reshaped_matrix.shape}")
            U, S, VT = svds(reshaped_matrix, k=200)
            np.save(os.path.join(svd_folder, f"{chemical_name}_U.npy"), U)
            np.save(os.path.join(svd_folder,f"{chemical_name}_S.npy"), S)
            np.save(os.path.join(svd_folder,f"{chemical_name}_V.npy"), VT)
            print("done!")

folder_path = "/home/nia/Desktop/extracted_chemicals/"
twohundred_mode_SVD(folder_path)

folder_path = '/home/nia/Desktop/extracted_chemicals/'

for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        print(f"File: {filename}")
        print(f"Shape of the data: {data.shape}")
        print("-" * 40)