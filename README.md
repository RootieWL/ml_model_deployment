# MLP Pipeline

The following demonstrates a simple end-to-end machine learning pipeline, it ingests/process the dataset and its able to perform univariate or multivariate LSTM Time Series Forecasting with TensorFlow. User is able to select the relevant target attribute, he/she wants to predict. (For eg total_number_of_escooters)
```
├── mlp                            # ML pipeline folder
│   ├── img                        # image folder where plots are generated
│   ├── extract_preprocess.py      # Extracts and Preprocess data
│   ├── rnn_tensorflow.py          # LSTM forecasting
│   └── params.py                  # Parameter file for various configuration
├── data_extraction.py      # 1. Data Extraction file
├── eda.ipynb               # 2. EDA
├── requirements.txt
├── run.sh                  
└── README.md
```

## Installation

Create a Conda venv environment with Python Version 3.6.8 and activate it, install the relevant dependencies. 

```bash
conda create --prefix venv python==3.6.8
conda activate venv/
pip install -r requirements.txt
```

## Usage
Execute run.sh with bash command in the following steps

```bash
bash run.sh extract_preprocess # 1. to extract and preprocess data for modeling
bash run.sh tf_lstm univariate total_scooter_users # 2.1 run either univariate/multivariate LSTM with target attribute
bash run.sh tf_lstm multivariate windspeed # 2.2 plots will be generated in mlp/img folder
```
