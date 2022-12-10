import os
import numpy as np
from sklearn.neighbors import KDTree as TUTUTree
import tqdm

target_path = "/home/csxjiang/jx/sync/mmclassification_dev/softmax_dump"
matched = {}
ID_dir = "ID"
OOD_dir = ["iNaturalist", "Places", "SUN", "Textures"]
full_samples = []
full_names = []
full_types = []
for sample in tqdm.tqdm(os.listdir(os.path.join(target_path, ID_dir))):
    f = open(os.path.join(target_path, ID_dir, sample), "r")
    with f:
        softmax_score = np.array(eval(f.readline().strip()))
        full_samples.append(softmax_score)
        full_names.append(os.path.splitext(sample)[0])
        full_types.append("ID")
OOD_samples = []
for OOD_subdir in OOD_dir:
    for sample in tqdm.tqdm(os.listdir(os.path.join(target_path, OOD_subdir))):
        f = open(os.path.join(target_path, OOD_subdir, sample), "r")
        with f:
            softmax_score = np.array(eval(f.readline().strip()))
            OOD_samples.append(os.path.splitext(sample)[0])
            full_samples.append(softmax_score)
            full_types.append("OOD")

tutu = TUTUTree(full_samples, metric="l1")
eps = 1e-3

for sample, name, type in zip(full_samples, full_names, full_types):
    if name in matched:
        continue
    dist, ind = tutu.query(sample, k=1)
    dist = dist[0]
    ind = ind[0]
    if dist < eps:
        print("({}){} <==> ({}){}, dist={}.".format(type, name, full_types[ind], full_names[ind], dist))
        matched[name] = None