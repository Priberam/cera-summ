from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI

from centroid_attention.model import LitCentroidEstimationModel
from centroid_attention.model_datasets import MyDataModuleBaseClass


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "model.batch_size",
            "data.init_args.batch_size",
        )
        parser.link_arguments(
            "model.num_positional_embeddings",
            "data.init_args.num_positional_embeddings",
        )

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({"early_stopping.monitor": "val_loss"})
        parser.set_defaults({"early_stopping.mode": "min"})
        parser.set_defaults({"early_stopping.patience": 5})

        parser.add_lightning_class_args(ModelCheckpoint, "val_loss_checkpoint")
        parser.set_defaults({"val_loss_checkpoint.monitor": "val_loss"})
        parser.set_defaults({"val_loss_checkpoint.mode": "min"})
        parser.set_defaults({"val_loss_checkpoint.save_top_k": 1})
        parser.set_defaults(
            {"val_loss_checkpoint.filename": ("{epoch}-{val_loss:.6f}")}
        )
        parser.add_lightning_class_args(ModelCheckpoint, "R2_R_checkpoint")
        parser.set_defaults({"R2_R_checkpoint.monitor": "R2_R"})
        parser.set_defaults({"R2_R_checkpoint.mode": "max"})
        parser.set_defaults({"R2_R_checkpoint.save_top_k": 1})
        parser.set_defaults(
            {"R2_R_checkpoint.filename": ("{epoch}-{R2_R:.6f}")}
        )


if __name__ == "__main__":
    cli = MyLightningCLI(
        LitCentroidEstimationModel,
        MyDataModuleBaseClass,
        subclass_mode_data=True,
        save_config_callback=None,
    )
