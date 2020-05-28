from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

from module import TPN2FModule
from tpn2f.model import TPN2FModel

if __name__ == "__main__":
    seed_everything(42)

    parser = ArgumentParser()
    parser = TPN2FModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(max_epochs=60, learning_rate=0.00115, num_workers=0, data_path='.data', dataset='MathQA')

    hparams = parser.parse_args()
    trainer = Trainer.from_argparse_args(hparams, deterministic=True)
    tpn2f = TPN2FModule(hparams)
    trainer.fit(tpn2f)
