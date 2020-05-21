from torch import empty, einsum
from torch.nn import Module, LSTMCell, Linear, Parameter
from torch.nn.functional import softmax


class TPN2FEncoderCell(Module):
    def __init__(self, hparams):
        super().__init__()
        self.hidden_size = hparams.role_size * hparams.filler_size
        self.temperature = hparams.temperature

        self.zeroth_filler_cell_state = Parameter(empty(1, self.hidden_size).zero_())
        self.filler_cell = LSTMCell(hparams.embedding_dim, self.hidden_size)

        self.filler_hidden_to_number = Linear(self.hidden_size, hparams.filler_number, bias=False)
        self.filler_dictionary = Linear(hparams.filler_number, hparams.filler_size, bias=False)

        self.zeroth_role_cell_state = Parameter(empty(1, self.hidden_size).zero_())
        self.role_cell = LSTMCell(hparams.embedding_dim, self.hidden_size)

        self.role_hidden_to_number = Linear(self.hidden_size, hparams.role_number, bias=False)
        self.role_dictionary = Linear(hparams.role_number, hparams.role_size, bias=False)

        self.zeroth_hidden_state = Parameter(empty(self.hidden_size).zero_())

    def forward(self, token_embedding, tensor_product, hidden_cell_states=None):
        hidden_state = self.zeroth_hidden_state if tensor_product is None else tensor_product
        if hidden_cell_states is None:
            filler_cell_state, role_cell_state = self.zeroth_filler_cell_state, self.zeroth_role_cell_state
        else:
            filler_cell_state, role_cell_state = hidden_cell_states[0], hidden_cell_states[1]

        hidden_state = hidden_state.unsqueeze(0)

        filler_hidden_state, filler_cell_state = self.filler_cell(token_embedding, (hidden_state, filler_cell_state))
        role_hidden_state, role_cell_state = self.role_cell(token_embedding, (hidden_state, role_cell_state))

        filler_number = softmax(self.filler_hidden_to_number(filler_hidden_state) / self.temperature, 1)
        filler = self.filler_dictionary(filler_number)

        role_number = softmax(self.role_hidden_to_number(role_hidden_state) / self.temperature, 1)
        role = self.role_dictionary(role_number)

        tensor_product = einsum("ij,ik", filler, role).reshape(self.hidden_size)

        return tensor_product, (filler_cell_state, role_cell_state)
