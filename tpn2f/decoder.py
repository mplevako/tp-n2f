from torch import empty
from torch.nn import LSTMCell, Module, Parameter
from torch.nn.init import xavier_uniform_, calculate_gain

from tpn2f.attention import Attention
from tpn2f.unbinding import UnbindingModule


class TPN2FDecoderCell(Module):
    def __init__(self, hparams):
        super().__init__()

        self.input_size = (1 + hparams.position_number) * hparams.embedding_dim
        self.hidden_size = hparams.argument_size * hparams.relation_size * hparams.position_size

        self.zeroth_tuple = Parameter(empty(1, self.input_size))

        self.zeroth_cell_state = Parameter(empty(1, self.hidden_size).zero_())
        self.cell = LSTMCell(self.input_size, self.hidden_size)

        self.attention = Attention(hparams)
        self.unbinding_module = UnbindingModule(hparams)

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.zeroth_tuple.data, calculate_gain('tanh'))

    def forward(self, context_vector, hidden_state, cell_state=None, relation_tuple=None):
        if relation_tuple is None:
            relation_tuple = self.zeroth_tuple

        cell_state = self.zeroth_cell_state if cell_state is None else cell_state

        hidden_state.unsqueeze(0)
        hidden_state, cell_state = self.cell(relation_tuple, (hidden_state, cell_state))

        hidden_state = self.attention(hidden_state, context_vector)
        relation_tuple = self.unbinding_module(hidden_state)

        return relation_tuple, hidden_state, cell_state
