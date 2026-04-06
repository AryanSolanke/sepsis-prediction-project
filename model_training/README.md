# Model Training Scripts

This folder now contains both the tree-based training scripts and three deep learning alternatives:

- `nn_model.py`: deep feedforward neural network for row-level tabular prediction
- `cnn_model.py`: 1D temporal CNN over patient windows
- `rnn_model.py`: GRU-based recurrent neural network over patient windows

All three scripts:

- read `Datasets/processed/sepsis_icu_cleaned.csv`
- split by `Patient_ID` to reduce leakage
- save trained artifacts to `model_training/models`
- write metrics and training history alongside the model checkpoint
- tune the final decision threshold on the validation set instead of blindly using `0.50`

## Why Precision Can Be Low

This dataset is heavily imbalanced, so a model can get high recall by predicting too many positives.
That usually hurts precision.

The updated scripts try to control that by:

- using focal loss by default
- using a softer class-weight strategy (`sqrt`) instead of the full imbalance ratio
- optionally enabling balanced sampling
- choosing a threshold that maximizes precision while keeping recall above a target

## Run From Repo Root

```powershell
python model_training\nn_model.py
python model_training\cnn_model.py
python model_training\rnn_model.py
```

Useful options:

```powershell
python model_training\nn_model.py --epochs 20 --batch-size 1024
python model_training\cnn_model.py --window-size 12 --epochs 12 --batch-size 256
python model_training\rnn_model.py --window-size 12 --epochs 12 --hidden-size 128
```

Recommended precision/recall-oriented runs:

```powershell
python model_training\nn_model.py --loss focal --class-weight-strategy sqrt --threshold-objective precision_at_recall --target-recall 0.70
python model_training\cnn_model.py --loss focal --class-weight-strategy sqrt --threshold-objective precision_at_recall --target-recall 0.70 --balanced-sampling
python model_training\rnn_model.py --loss focal --class-weight-strategy sqrt --threshold-objective precision_at_recall --target-recall 0.70 --balanced-sampling
```

## Saved Artifacts

Feedforward NN:

- `model_training/models/sepsis_nn.pt`
- `model_training/models/nn_feature_names.joblib`
- `model_training/models/nn_scaler.joblib`
- `model_training/models/nn_metrics.json`
- `model_training/models/nn_history.json`

Temporal CNN:

- `model_training/models/sepsis_cnn.pt`
- `model_training/models/cnn_feature_names.joblib`
- `model_training/models/cnn_scaler.joblib`
- `model_training/models/cnn_metrics.json`
- `model_training/models/cnn_history.json`

Recurrent NN:

- `model_training/models/sepsis_rnn.pt`
- `model_training/models/rnn_feature_names.joblib`
- `model_training/models/rnn_scaler.joblib`
- `model_training/models/rnn_metrics.json`
- `model_training/models/rnn_history.json`

## Dependency Note

The deep learning scripts use PyTorch. If `torch` is not installed in your environment, the scripts will exit with a clear message before training starts.
