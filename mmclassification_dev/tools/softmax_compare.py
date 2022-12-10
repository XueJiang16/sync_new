import os
import numpy as np
from sklearn.neighbors import KDTree as TUTUTree
import tqdm

target_path = "/home/csxjiang/jx/sync/mmclassification_dev/softmax_dump"
ID_dir = "ID"
OOD_dir = ["iNaturalist", "Places", "SUN", "Textures"]
ID_samples = []
for sample in tqdm.tqdm(os.listdir(ID_dir)):
    f = open(os.path.join(target_path, ID_dir, sample), "r")
    with f:
        ID_samples.append(np.array(eval(f.readline().strip())))
    print(ID_samples[0])
    assert False
OOD_samples = []
for OOD_subdir in OOD_dir:
    for sample in tqdm.tqdm(os.listdir(OOD_subdir)):
        f = open(os.path.join(target_path, OOD_subdir, sample), "r")
        with f:
            OOD_samples.append(np.array(eval(f.readline().strip())))
