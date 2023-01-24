import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter
import tqdm
import seaborn as sns
import os

def single_picture(basic_path, method, dataset, mode=''):
    path = basic_path
    in_confs = np.load(path+'/ID.npy')
    in_confs = in_confs[:,0]
    out_confs = np.load(path+'/{}.npy'.format(dataset))
    out_confs = out_confs[:,0]
    #id-ood
    sns.set(rc={'figure.figsize': (8, 6)})
    # sns.set_style('whitegrid')
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    fig = sns.kdeplot(np.array(in_confs), bw=0.2)
    sns.kdeplot(np.array(out_confs), bw=0.2)
    plt.legend(labels=['ID (ImageNet)', 'OOD ({})'.format(dataset)])
    plt.xlabel('OOD scores', fontsize=18, fontweight='bold')
    plt.ylabel('Density', fontsize=18, fontweight='bold')
    plt.yticks(size=16)
    plt.xticks(size=16)
    plt.tight_layout()
    fig.get_figure().savefig(os.path.join(basic_path, '{}_overall.jpg'.format(dataset)))
    plt.close()

if __name__ == '__main__':
    basic_path = './feature_sim_dump'
    # methods = ['MSP','ODIN','Energy']
    # methods = ['GradNorm']
    methods = ['ours']
    datasets = ['iNaturalist','SUN','Places','Textures']
    for method in methods:
        for dataset in datasets:
            single_picture(basic_path, method, dataset)
