# FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection
![framework](imgs/framework.png)
This repository holds the code for FraudGT framework.

## Environment Setup
You can create a conda environment to easily run the code. For example, we can create a virtual environment named `fraudGT`:
```
conda create -n fraudGT python=3.10 -y
conda activate fraudGT
```
Install the required packages using the following commands:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```

## Run the Code
You will need to firstly specify the dataset path (`./data` in this example) and log location (`./results` in this example) by editing the config file provided under `./configs/{dataset_name}/`. An example configuration is
```
......
out_dir: ./results
dataset:
  dir: ./data
......
```
Download and unzip the [Anti-Money Laundering dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) into your specified dataset path (for example, put the unzipped `HI-small.csv` into `./data`).
Dataset will be automatically processed at the first run.

![experiments](imgs/experiment.png)

For convenience, a script file is created to run the experiment with specified configuration. For instance, you can edit and run the `interactive_run.sh` to start the experiment.
```
cd FraudGT
chmox +x ./run/interactive_run.sh
./run/interactive_run.sh
```

## Heterogeneous Graph Support

This repository includes modifications to support heterogeneous graph learning on the Elliptic++ dataset with transaction (tx) and wallet (address) node types. 

### Key Features
- **EllipticPPPyG_TxWallet**: New dataset class for tx+wallet heterogeneous graphs
- **Target Node Type Classification**: Support for node type-specific classification (e.g., classify only 'tx' nodes)
- **HGT Model Integration**: Modified HGT model to work with heterogeneous graphs

### Quick Start
See `REPRODUCTION_GUIDE.md` and `HETERO_GRAPH_GUIDE.md` for detailed instructions on:
- Setting up the dataset
- Running heterogeneous graph experiments
- Understanding the modifications made to FraudGT

### Modified Files
- `fraudGT/datasets/ellipticpp_tx_wallet_pyg.py` - New dataset class
- `fraudGT/config/dataset_config.py` - Added `target_ntype` configuration
- `fraudGT/loader/master_loader.py` - Added dataset loader hook
- `fraudGT/graphgym/loader.py` - Support for target node type
- `fraudGT/graphgym/models/hgt.py` - HGT model modifications
- `configs/ellipticpp-txwallet-hgt.yaml` - Configuration file

