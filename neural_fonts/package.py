import glob
import os
import pickle
import random

import click


def pickle_examples(
    paths,
    train_path: str,
    val_path: str,
    train_val_split: float = 0.1,
    fixed_sample: bool = False,
):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    if fixed_sample:
        with open(train_path, "wb") as ft:
            with open(val_path, "wb") as fv:
                for p in paths:
                    label = int(os.path.basename(p).split("_")[0])
                    uni = os.path.basename(p).split("_")[1]
                    with open(p, "rb") as f:
                        #                        print("img %s" % p, label)
                        img_bytes = f.read()
                        example = (label, uni, img_bytes)
                        if "val" in p:
                            #                            print("img %s is saved in val.obj" % p)
                            # validation set
                            pickle.dump(example, fv)
                        else:
                            # training set
                            #                            print("img %s is saved in train.obj" % p)
                            pickle.dump(example, ft)
                return
    with open(train_path, "wb") as ft:
        with open(val_path, "wb") as fv:
            for p in paths:
                label = int(os.path.basename(p).split("_")[0])
                with open(p, "rb") as f:
                    print("img %s" % p, label)
                    img_bytes = f.read()
                    r = random.random()
                    example = (label, img_bytes)
                    if r < train_val_split:
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)


@click.command()
@click.option(
    "--dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    required=True,
    help="path of examples",
)
@click.option(
    "--save-dir", type=click.Path(), required=True, help="path to save pickled files"
)
@click.option(
    "--split-ratio", type=float, default=0.1, help="split ratio between train and val"
)
@click.option(
    "--fixed-sample",
    type=bool,
    default=False,
    help="binarize fixed samples (we distinguish train/validation data with its filename).",
)
def main(dir: str, save_dir: str, split_ratio: float, fixed_sample: bool):
    """
    Compile list of images into a pickled object for training
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_path = os.path.join(save_dir, "train.obj")
    val_path = os.path.join(save_dir, "val.obj")
    pickle_examples(
        glob.glob(os.path.join(dir, "*.png")),
        train_path=train_path,
        val_path=val_path,
        train_val_split=split_ratio,
        fixed_sample=fixed_sample,
    )

    """ pickle_examples(
        sorted(
            glob.glob(os.path.join(dir, "*.png")),
            key=lambda e: float(
                os.path.splitext(os.path.basename(e))[0]
                .replace("_", "")
                .replace("train", "")
                .replace("val", "")
            ),
        ),
        train_path=train_path,
        val_path=val_path,
        train_val_split=split_ratio,
        fixed_sample=fixed_sample,
    ) """
