from __future__ import print_function, division, absolute_import

# Data wrangling imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# Utility imports
import os
import os.path
import time
import ast
import copy
from datetime import datetime
from tqdm import tqdm

# Data visualization imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import seaborn as sns

# PyTorch imports
import torch

# Feature generator
from feature import LANL_FeatureGenerator


SEED_VAL = 42


def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    print(torch.cuda.get_device_name(device))

    torch.backends.cudnn.benchmark = True

    # Setting the seeds for reproducibility
    np.random.seed(SEED_VAL)
    if use_cuda:
        torch.cuda.manual_seed_all(SEED_VAL)
    else:
        torch.manual_seed_all(SEED_VAL)

    test_df = LANL_FeatureGenerator(dtype='test', n_jobs=16, chunk_size=None)
    test_df = test_df.generate()
    predictions = pd.read_csv('../submissions/submission_20190423-161737.csv')
    merged_pred = pd.merge(
        predictions,
        test_df,
        on='seg_id').drop(
            'target',
        axis=1)

    merged_pred.head()

    for row in tqdm(merged_pred.itertuples(index=False)):
        test_plot(row)


def test_plot(seg):
    fig, ax = plt.subplots()

    ax.set_xlabel('Time')
    ax.set_ylabel('Acoustic Data')
    ax.set_ylim(-200, 200)
    ax.set_title(seg.seg_id + '\nPredicted Time-to-Failure: ' +
                 str(seg.time_to_failure))
    ax.plot(seg.segment, 'r')

    plt.savefig('../static/img/' + seg.seg_id + '.png')
    plt.close()
