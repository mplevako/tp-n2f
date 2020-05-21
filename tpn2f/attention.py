from torch import cat, matmul
from torch.nn import Module, Linear
from torch.nn.functional import softmax


class Attention(Module):
    def __init__(self, hparams):
        super().__init__()
        self.query_size = hparams.argument_size * hparams.relation_size * hparams.position_size

        self.K = Linear(2 * self.query_size, self.query_size, bias=False)

    def forward(self, query, context_vector):
        query = query.squeeze()
        score = matmul(context_vector, query)
        attention = softmax(score, 0)
        s_t = matmul(attention, context_vector)
        gamma = cat([query, s_t])
        return self.K(gamma)
