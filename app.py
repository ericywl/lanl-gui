import base64
import re
import io
import json
from flask import Flask, render_template, request, jsonify

import torch
import pandas as pd
from src.model import DenseNet
from src.predict import LANL_pred


app = Flask(__name__)
model = DenseNet().to(torch.device("cpu"))
model.load_state_dict(torch.load(
    './snapshots/snapshot_20190423-144915_2.1862827587810507.pt'))

predictions = pd.read_csv('./submissions/submission_20190423-161737.csv')
predictions = {
    row.seg_id: row.time_to_failure for row in predictions.itertuples(index=False)
}

@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    if request.method == "POST":
        if not request.content_type == "application/json":
            return "", "400 REQUEST DATA NOT JSON"
        csv_data = re.sub("^data:text/csv;base64,", "",
                          request.get_json()["data"])
        fname = request.get_json()["name"]
        csv_blob = base64.b64decode(csv_data)
        output = LANL_pred(model, fname, io.BytesIO(csv_blob))
        return jsonify(output)
    return "", "400 INVALID REQUEST METHOD"


@app.route("/ranks")
def ranks():
    return render_template("ranks.html", data=predictions)


if __name__ == "__main__":
    app.run()
