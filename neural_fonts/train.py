import click
import tensorflow as tf

from neural_fonts.model.unet import UNet


@click.command()
@click.option(
    "--experiment-dir",
    type=click.Path(),
    required=True,
    help="experiment directory, data, samples, checkpoints, etc.",
)
@click.option(
    "--experiment-id",
    type=int,
    default=0,
    help="sequence id for the experiments you prepare to run",
)
@click.option(
    "--image-size", type=int, default=128, help="size of your input and output image"
)
@click.option("--L1-penalty", type=int, default=100, help="weight for L1 loss")
@click.option("--Lconst-penalty", type=int, default=15, help="weight for const loss")
@click.option("--Ltv-penalty", type=float, default=0.0, help="weight for tv loss")
@click.option(
    "--Lcategory-penalty", type=float, default=1.0, help="weight for category loss"
)
@click.option(
    "--embedding-num", type=int, default=40, help="number for distinct embeddings"
)
@click.option("--embedding-dim", type=int, default=128, help="dimension for embedding")
@click.option("--epoch", type=int, default=100, help="number of epoch")
@click.option("--batch-size", type=int, default=16, help="number of examples in batch")
@click.option("--lr", type=float, default=0.001, help="initial learning rate for adam")
@click.option(
    "--schedule", type=int, default=10, help="number of epochs to half learning rate"
)
@click.option("--resume", type=bool, default=True, help="resume from previous training")
@click.option(
    "--freeze-encoder",
    type=bool,
    default=False,
    help="freeze encoder weights during training",
)
@click.option(
    "--fine-tune",
    type=str | None,
    default=None,
    help="specific labels id to be fine tuned",
)
@click.option(
    "--inst-norm",
    type=bool,
    default=False,
    help="use conditional instance normalization in your model",
)
@click.option(
    "--sample-steps",
    type=int,
    default=10,
    help="number of batches in between two samples are drawn from validation set",
)
@click.option(
    "--checkpoint-steps",
    type=int,
    default=500,
    help="number of batches in between two checkpoints",
)
@click.option(
    "--flip-labels",
    type=bool,
    default=False,
    help="whether flip training data labels or not, in fine tuning",
)
@click.option("--no-val", type=bool, default=False, help="no validation set is given")
def main(
    experiment_dir: str,
    experiment_id: int,
    image_size: int,
    L1_penalty: int,
    Lconst_penalty: int,
    Ltv_penalty: float,
    Lcategory_penalty: float,
    embedding_num: int,
    embedding_dim: int,
    epoch: int,
    batch_size: int,
    lr: float,
    schedule: int,
    resume: bool,
    freeze_encoder: bool,
    fine_tune: str | None,
    inst_norm: bool,
    sample_steps: int,
    checkpoint_steps: int,
    flip_labels: bool,
    no_val: bool,
):
    """Train"""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        model = UNet(
            experiment_dir,
            batch_size=batch_size,
            experiment_id=experiment_id,
            input_width=image_size,
            output_width=image_size,
            embedding_num=embedding_num,
            embedding_dim=embedding_dim,
            L1_penalty=L1_penalty,
            Lconst_penalty=Lconst_penalty,
            Ltv_penalty=Ltv_penalty,
            Lcategory_penalty=Lcategory_penalty,
        )
        model.register_session(sess)
        if flip_labels:
            model.build_model(
                is_training=True, inst_norm=inst_norm, no_target_source=True
            )
        else:
            model.build_model(is_training=True, inst_norm=inst_norm)
        fine_tune_list: set[int] | None = None
        if fine_tune is not None:
            fine_tune_list = {int(i) for i in fine_tune.split(",")}
        model.train(
            lr=lr,
            epoch=epoch,
            resume=resume,
            schedule=schedule,
            freeze_encoder=freeze_encoder,
            fine_tune=fine_tune_list,
            sample_steps=sample_steps,
            checkpoint_steps=checkpoint_steps,
            flip_labels=flip_labels,
            no_val=no_val,
        )
