from argparse import ArgumentParser

from torch import argmax, stack, sum, cat
from torch.nn import Embedding

from tpn2f.decoder import TPN2FDecoderCell
from tpn2f.encoder import TPN2FEncoderCell
from tpn2f.mapper import ReasoningModule


class TPN2FModel:
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder_embeddings = Embedding(hparams.encoder_num_embeddings, hparams.embedding_dim)
        self.encoder_cell = TPN2FEncoderCell(hparams)
        self.mapper = ReasoningModule(hparams)
        self.decoder_relation_embeddings = Embedding(hparams.relation_number, hparams.embedding_dim)
        self.decoder_argument_embeddings = Embedding(hparams.argument_number, hparams.embedding_dim)
        self.decoder_cell = TPN2FDecoderCell(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--temperature', type=float, default=0.1)
        parser.add_argument('--embedding_dim', type=int, default=100)
        parser.add_argument('--role_number', type=int, default=50)
        parser.add_argument('--role_size', type=int, default=30)
        parser.add_argument('--filler_number', type=int, default=50)
        parser.add_argument('--filler_size', type=int, default=30)
        parser.add_argument('--relation_size', type=int, default=30)
        parser.add_argument('--argument_size', type=int, default=20)
        parser.add_argument('--position_size', type=int, default=5)
        return parser

    def forward(self, tokens, max_answer_length, formula=None):
        relations, arguments = [], []
        hidden_state, context_vector = self.encode(tokens)
        cell_state, relation_tuple = None, None
        for time in range(max_answer_length):
            hidden_state = hidden_state.unsqueeze(0)
            relation_tuple, hidden_state, cell_state = self.decoder_cell(context_vector, hidden_state, cell_state,
                                                                         relation_tuple)
            relations.append(relation_tuple[0])
            arguments.append(relation_tuple[1])
            relation_tuple = formula[time] if formula is not None else self.decode_relation(relation_tuple)
            relation_tuple = cat((self.decoder_relation_embeddings(relation_tuple[0]).unsqueeze(0),
                                  self.decoder_argument_embeddings(relation_tuple[1:]))).reshape(-1).unsqueeze(0)
        return stack(relations), stack(arguments)

    def decode_formula(self, formula):
        return [self.decode_relation((relation, args)) for relation, args in zip(formula[0], formula[1])]

    @staticmethod
    def decode_relation(relation_args):
        return cat([argmax(relation_args[0]).unsqueeze(0), argmax(relation_args[1], 0)])

    def encode(self, tokens):
        tensor_products = []
        tensor_product, hidden_cell_states = None, None
        for token_embedding in self.encoder_embeddings(tokens.squeeze()):
            tensor_product, hidden_cell_states = self.encoder_cell(token_embedding.unsqueeze(0), tensor_product,
                                                                   hidden_cell_states)
            tensor_products.append(tensor_product)
        context_vector = stack(tensor_products)
        tpr_embedding = sum(context_vector, 0)
        decoder_initial_hidden_state, context_vector = self.mapper(tpr_embedding, context_vector)
        return decoder_initial_hidden_state, context_vector
