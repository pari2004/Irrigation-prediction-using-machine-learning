# Hybrid Irrigation Prediction Model

This project implements a hybrid irrigation prediction model that combines a physics-based model with a machine learning model to provide accurate and reliable irrigation recommendations.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate sample data:**

    ```bash
    python train_model.py --generate-data
    ```

## Training

To train the model, run the following command:

```bash
python train_model.py --model-type [xgboost|lightgbm]
```

To compare the performance of different models, run:

```bash
python train_model.py --compare-models
```

## Prediction

The prediction logic is integrated into the training script and is executed automatically after the model is trained. The example predictions are displayed in the console.
