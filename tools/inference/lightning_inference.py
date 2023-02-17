"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to image(s) to infer.")
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    args = parser.parse_args()
    return args


def infer():
    """Run inference."""
    args = get_args()
    config = get_configurable_parameters(config_path=args.config)
    config.trainer.resume_from_checkpoint = str(args.weights)
    config.visualization.show_images = args.show
    config.visualization.mode = args.visualization_mode
    if args.output:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = args.output
    else:
        config.visualization.save_images = False

    model = get_model(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(callbacks=callbacks, **config.trainer)

    transform_config = config.dataset.transform_config.val if "transform_config" in config.dataset.keys() else None
    dataset = InferenceDataset(
        config.dataset.path, args.input, image_size=tuple(config.dataset.image_size), transform_config=transform_config
    )
    dataloader = DataLoader(dataset)
    pred = trainer.predict(model=model, dataloaders=[dataloader])

    # summarize results
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    pred_data = []
    for results in pred:
        pred_data.append({
            'image_path': results['image_path'][0],
            # 'map': results['anomaly_maps'][0].tolist(),
            'score': results['pred_scores'][0].tolist(),
            'prediction': results['pred_labels'][0].tolist(),
            # 'mask': results['pred_masks'][0].tolist(),
        })
    df_pred = pd.DataFrame(pred_data)
    # ground truth - positive class = True (anomalous)
    df_pred['installation'] = df_pred.image_path.str.contains('bad')
    cm = confusion_matrix(df_pred['installation'], df_pred['prediction'])
    print(cm)
    cm_norm = confusion_matrix(df_pred['installation'], df_pred['prediction'], normalize='true') * 100
    print(cm_norm)
    # df_pred.to_parquet(args.output + '/results.pq')
    df_pred.to_csv(args.output + '/results.csv')


if __name__ == "__main__":
    infer()
