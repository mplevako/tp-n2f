from torch import tanh
from torch.nn import Module, Linear


class ReasoningModule(Module):
    def __init__(self, hparams):
        super().__init__()

        self.encoder_hidden_size = hparams.role_size * hparams.filler_size
        self.decoder_hidden_size = hparams.argument_size * hparams.relation_size * hparams.position_size
        self.structure_mapper = Linear(self.encoder_hidden_size, self.decoder_hidden_size, bias=False)

    def forward(self, tpr_embedding, context_vector):
        return tanh(self.structure_mapper(tpr_embedding)), self.structure_mapper(context_vector)
