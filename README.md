# Pascal VOC Deep Learning Project

## Setup

Install a few things before running the code, preferably in a new
Python virtual environment.

```
pip install -r requirements.txt
```

## Web Gui

Run `python app.py` and head to `localhost:5000` on a browser.

## Running other stuff
You will need to download the dataset from https://www.kaggle.com/c/LANL-Earthquake-Prediction/data.

Then, do the following:
```
mkdir data
mv LANL-Earthquake-Prediction.zip data/
cd data/
unzip LANL-Earthquake-Prediction.zip
chmod 644 *

mkdir test
mv test.zip test/
cd test/
unzip test.zip
chmod 644 *
```