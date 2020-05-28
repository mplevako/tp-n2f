from pytorch_lightning import LightningModule
from torch import argmax, stack, tensor, zeros, bitwise_and
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader

from tpn2f.model import TPN2FModel


class TPN2FModule(TPN2FModel, LightningModule):
    def __init__(self, hparams):
        import mathqa
        self.hparams = hparams
        self.dataset = getattr(mathqa, hparams.dataset)
        self.dataset.build_vocabs(hparams.data_path)
        hparams.position_number = self.dataset.arguments.fix_length
        hparams.relation_number = len(self.dataset.relation.vocab.itos)
        hparams.argument_number = len(self.dataset.arguments.vocab.itos)
        hparams.encoder_num_embeddings = len(self.dataset.problem.vocab.itos)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        super().__init__(hparams)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def step(self, batch, prefix=None):
        loss = zeros(1, requires_grad=True)
        problem, ground_truth_formula = batch[0].squeeze(), batch[1].squeeze(0)
        relation_tuples = self.forward(problem, len(ground_truth_formula), ground_truth_formula)
        loss = loss + cross_entropy(relation_tuples[0], ground_truth_formula[:, 0])
        loss = loss + cross_entropy(relation_tuples[1], ground_truth_formula[:, 1:])
        op_acc = bitwise_and(argmax(relation_tuples[0], 1) == ground_truth_formula[:, 0],
                             (stack([argmax(x, 0) for x in relation_tuples[1]]) == ground_truth_formula[:, 1:]).all(1))
        op_acc = op_acc.type_as(relation_tuples[0]).mean()
        stats = {f'{prefix}_loss' if prefix else 'loss': loss, f'{prefix}_acc' if prefix else 'acc': float(op_acc)}
        return {**stats, 'log': stats}

    def epoch_end(self, outputs, prefix='val'):
        avg_loss = stack([output[f'{prefix}_loss'] for output in outputs]).mean()
        avg_operation_accuracy = tensor([output[f'{prefix}_acc'] for output in outputs]).mean()
        stats = {f'{prefix}_loss': avg_loss, f'{prefix}_acc': avg_operation_accuracy}
        return {**stats, 'log': stats, 'progress_bar': stats}

    def training_step(self, batch, batch_id):
        return self.step(batch)

    def validation_step(self, batch, batch_id):
        return self.step(batch, 'val')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs)

    def test_step(self, batch, batch_id):
        return self.step(batch, 'test')

    def test_epoch_end(self, outputs):
        return self.epoch_end(outputs)

    def prepare_data(self):
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset.splits(train=self.dataset.train_set,
                                                                                      validation=self.dataset.val_set,
                                                                                      test=self.dataset.train_set,
                                                                                      root=self.hparams.data_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.hparams.num_workers)
