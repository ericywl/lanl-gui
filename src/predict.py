from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')


def LANL_pred(model, fname, csv, device=None):
    if not device:
        device = torch.device("cpu")

    in_df = pd.read_csv(csv)
    scaler = joblib.load('./helper/scaler.joblib')
    scaled_in = scaler.transform(
        in_df.acoustic_data.values.astype(float).reshape(-1, 1)).flatten()

    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(scaled_in).unsqueeze(
            0).unsqueeze(0).float().to(device))

    fig, ax = plt.subplots()

    ax.set_xlabel('Time')
    ax.set_ylabel('Acoustic Data')
    ax.set_ylim(-200, 200)
    ax.set_title('Predicted Time-to-Failure: ' + str(output.item()))
    ax.plot(in_df.acoustic_data.values, 'r')

    img_name = os.path.splitext(fname)[0] + '.png'
    plt.savefig('./static/img/preds/' + img_name)
    plt.close()
    return {
        "time-to-failure": str(output.item()),
        "filename": img_name
    }
