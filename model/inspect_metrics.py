#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

LIGHTNING_LOGS = Path(__file__).parent.parent.joinpath('lightning_logs')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--version',
        type=int,
        default=None,
        help='Training version to inspect'
    )
    return parser.parse_args()


def get_version_data(version: int = None):
    if version is None:
        latest_version = max(
            (d for d in LIGHTNING_LOGS.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime
        )
        version_path = latest_version
    else:
        version_path = LIGHTNING_LOGS.joinpath(f'version_{version}')
    return version_path


def draw_progress(df: pd.DataFrame):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs = axs.ravel()

    fig.suptitle('Model training metrics')

    axs[0].set_title('Accuracy')
    plot_epoch(axs[0], df, 'train/acc', 'train')
    plot_epoch(axs[0], df, 'val/acc', 'val')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title('Loss')
    plot_epoch(axs[1], df, 'train/loss', 'train')
    plot_epoch(axs[1], df, 'val/loss', 'val')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid()

    return fig


def plot_epoch(ax: plt.Axes, df: pd.DataFrame, col: str, label: str):
    key = 'epoch'
    data = df.loc[df[col].notna(), [key, col]].sort_values(key)
    ax.plot(data[key], data[col], label=label)


def main(version: int = None):
    version_path = get_version_data(version)
    if not version_path.exists():
        print(f'Directory does not exist: {version_path}')
        return
    df = pd.read_csv(version_path.joinpath('metrics.csv'), sep=',')
    fig = draw_progress(df)
    fig.savefig(version_path.joinpath('metrics.png'))
    plt.show()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.version)
