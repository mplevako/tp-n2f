from torch import einsum, empty, matmul, sum
from torch.nn import Module, Parameter
from torch.nn.functional import softmax
from torch.nn.init import calculate_gain, xavier_uniform_


class UnbindingModule(Module):
    def __init__(self, hparams):
        super().__init__()
        self.beta = 1e2
        self.argument_size = hparams.argument_size
        self.relation_size = hparams.relation_size
        self.position_size = hparams.position_size
        self.positions = Parameter(empty(hparams.position_size, hparams.position_number))
        self.relation_unbinder = Parameter(empty(hparams.relation_size, hparams.argument_size * hparams.relation_size))
        self.relation_classifier = Parameter(empty(hparams.relation_number, hparams.relation_size))
        self.argument_classifier = Parameter(empty(hparams.argument_number, hparams.argument_size))

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.positions.data, calculate_gain('tanh'))
        xavier_uniform_(self.relation_unbinder.data, calculate_gain('tanh'))
        xavier_uniform_(self.relation_classifier.data, calculate_gain('tanh'))
        xavier_uniform_(self.argument_classifier.data, calculate_gain('tanh'))

    def forward(self, hidden_state):
        hidden_state = hidden_state.reshape(self.argument_size, self.relation_size, self.position_size)
        op_arg_bindings = matmul(hidden_state, self.positions)
        relation_unbinding = matmul(self.relation_unbinder, sum(op_arg_bindings, 2).reshape(-1))
        arg_embeddings = einsum("ijk,j", op_arg_bindings, relation_unbinding)
        relation_classification = matmul(self.relation_classifier, relation_unbinding)
        arg_classifications = matmul(self.argument_classifier, arg_embeddings)

        args, relation = self.softargmax(arg_classifications, relation_classification)

        return [relation, args]

    def softargmax(self, arg_classifications, relation_classification):
        relation = einsum("i,i->i", relation_classification, softmax(self.beta * relation_classification, 0))
        args = einsum("ij,ij->ij", arg_classifications, softmax(self.beta * arg_classifications, 0))
        return args, relation
