# RPS Model

A CNN trained using transfer learning with `MobileNetV2` as the backbone.

## Dataset

Dataset was downloaded from
[kaggle](https://www.kaggle.com/datasets/frtgnn/rock-paper-scissor) and shall
be extracted to `~/.kaggle/rock-paper-scissors`.

The following structure is assumed:

```
~/.kaggle
└─── rock-paper-scissors
    ├─── rps
    │   └─── rps
    │       ├─── paper
    │       ├─── rock
    │       └─── scissors
    └─── rps-test-set
        └─── rps-test-set
            ├─── paper
            ├─── rock
            └─── scissors
```

## Training

Hyperparameters such as batch size, learning rate and epochs can be set in
[train.py](train.py). Afterward, start training from the command line:

```bash
python -m model.train
```

After the successful training, the accuracy and loss per epoch can be inspected
using the logs written by `pytorch-lightning` in `lightning_logs`:

```bash
python -m model.inspect_metrics
```

The final model will be saved in `out/rps_mobilenetv2.pt`.

## Inference

After successful training, the model can be tested by running inference on the
three test images in the `res/images` folder:

```bash
python -m model.infer
```