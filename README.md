# Step by step guide

This is the analysis code to reproduce the results of the DPV manuscript "Leveraging pretrained large language model for prognosis of type 2 diabetes with longitudinal medical records" [LINK]. Please follow the steps below to run the analysis and the machine learning framework for similar health data. 

## Data
The data used in this manuscript was obtained from the Diabetes-Patienten-Verlaufsdokumentation (DPV), an initiative to enhance diabetes research and treatment. For more information about the project and data acquitition please visit: https://buster.zibmt.uni-ulm.de/.

Please clone this repository and obtain the data to your local computer. 

## Install conda environment
First install conda if you have not done that. Then within this repos directory install and activate the conda environment using:

```sh
conda env create -f dpv_environment.yml
conda activate DPV
```

## Preprocess raw data and make input
The raw data obtained from DPV was a `.csv` file with columns representing clinical variables and rows representing clinic visits grouped by patients. To preprocess the raw data and create the input for the machine learning framework, first modify the parameters in the `/create_input/config.yml` to suit your analysis and run the following command:

```sh
python create_input.py
```

Note that the function `feature_recode()` in the `/create_input/preprocessing.py` script contains many hard coded processing, modify the function to suit your analysis. 

## Generate synthetic data
Due to specific circumstances we needed to generate synthetic data from the original DPV data, and used both datasets in parallel throughout the analysis. Here we implemented the variational autoencoder on top of the long short term memory modeling (LSTM-VAE) to robustly preserving the original data distribution and the autoregressive data structure while introducing enough noise to make new data. 

To generate synthetic data, simply edit the `generate_synthetic_data/config.yml` file and then run the following script:

```sh
python generate_synthetic_data.py
```
## Make prediction leveraging pretrained large language models
Medical time series data is notably very sparse with a lot of missing information. Futhermore, high number of features complicates the analyses. In order to leverage the prediction power of pretrained large language models in the most efficient way, I introduce a novel data processing and embedding method to bridge the gap between the two domains. Details of the method could be found in the manuscript [LINK]. In short, missing information is summarized as a binary table which is appended to the training data as additional features. In addition, a learnable embedding layer is prepended to the pretrained LLM to adapt the numeric data to the LLM architecture. Both components are learned simultaneously during training. 

![The machine learning framework to leverage pretrained LLM for medical timeseries data](workflow.png)

To run the framework, just modify the `config_run_models.yml` file to set the parameters for the model (as well as LSTM and XGBoost), then run the following command:

```sh
python run_transformer.py
```

In the manuscript we also benchmarked the performance of our model with LSTM and XGBoost. Run the two algorithms (after setting necessary parameters) with the following commands:

```sh
python run_lstm.py
python run_xgb.py
```
### Cross-validation
In the manuscript we did several benchmarking by cross-validation. To run the cross-validation, please set the neccessary parameters by adjusting the `config_cv.yml` file, wherever applicable. 

To compare the performance of our proposed model and the conventional LSTM at different training sizes, run the following commands, respectively:

```sh
python cv_llm.py
python cv_lstm.py
```

You can also compare the performance of our model versus XGBoost, at different prediction points, by running the following commands, respectively:

```sh
python cv_llm_slidWindows.py
python cv_xgb_slidWindows.py
```