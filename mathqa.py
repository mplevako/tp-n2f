import re
from io import open
from json import load
from os.path import expanduser, join

from torch import tensor
from torchtext.data import Field, Dataset


class MathQA(Dataset):
    urls = ('https://math-qa.github.io/math-QA/data/MathQA.zip',)
    encoding = 'utf8'
    name = 'MathQA'
    dirname = ''

    relation_list = 'operation_list.txt'
    constant_list = 'constant_list.txt'
    train_set = 'train.json'
    val_set = 'dev.json'
    test_set = 'test.json'

    relation_constant_tokenizer = re.compile('\\W+')
    learnt_constant_reference = re.compile('n\\d+')
    formula_step_separator = '|'
    formula_step_tokenizer = re.compile('[(),|]')

    problem_key = 'Problem'
    formula_key = 'linear_formula'

    problem, relation, arguments = Field(lower=True, pad_first=False), Field(), Field()

    @classmethod
    def build_vocabs(cls, data_path):
        path = cls.download(expanduser(data_path))
        relations, arguments = relation_args_whitelists = [[], []]
        for i, whitelist_file in enumerate([cls.relation_list, cls.constant_list]):
            with open(join(path, whitelist_file), encoding=cls.encoding) as reader:
                lines = reader.readlines()
                relation_args_whitelists[i].extend([[s.rstrip('\n')] for s in lines])

        max_args = 0
        problem_num = 0
        problem_tokens = []
        for dataset_file in [cls.train_set, cls.val_set, cls.test_set]:
            with open(join(path, dataset_file), encoding=cls.encoding) as reader:
                for json in load(reader):
                    problem, formula = json[cls.problem_key], json[cls.formula_key]
                    problem_num += 1
                    problem_tokens.extend(cls.problem.preprocess(problem))
                    for rel, *args in cls.tokenize_formula(formula):
                        relations.append(rel)
                        arguments.extend(args)
                        max_args = max(max_args, len(args))
        cls.problem.build_vocab([problem_tokens])
        cls.relation.build_vocab([relations])
        cls.arguments.build_vocab([arguments])
        cls.arguments.fix_length = max_args
        cls.problem_num = problem_num
        cls.arg_pad_idx = cls.arguments.vocab.stoi[cls.arguments.pad_token]

    @classmethod
    def tokenize_formula(cls, formula):
        return [[x for x in ra] for ra in [MathQA.formula_step_tokenizer.split(step)[:-1] for step in
                                           formula.rstrip(MathQA.formula_step_separator).split(
                                               MathQA.formula_step_separator)]]

    @classmethod
    def encode_problem(cls, problem):
        return cls.problem.process([cls.problem.preprocess(problem)]).T

    @classmethod
    def encode_formula(cls, formula):
        encoded_formula = []
        tokenized_formula = cls.tokenize_formula(formula)
        for relation in tokenized_formula:
            processed_relation = [cls.relation.vocab.stoi[relation[0][0]]]
            processed_relation.extend([cls.arguments.vocab.stoi[arg[0]] for arg in relation[1:]])
            processed_relation.extend([cls.arg_pad_idx] * (cls.arguments.fix_length - len(relation) + 1))
            encoded_formula.append(processed_relation)
        return tensor(encoded_formula)

    @classmethod
    def translate_formula(cls, formula):
        return [[cls.relation.vocab.itos[rel_args[0]]] + [cls.arguments.vocab.itos[arg] for arg in rel_args[1:] if
                                                          arg != cls.arg_pad_idx] for rel_args in formula]

    def __init__(self, path, **kwargs):
        examples = []
        fields = {'problem': self.problem, ('relation', 'arguments'): (self.relation, self.arguments)}
        with open(path, encoding=self.encoding) as reader:
            for json in load(reader):
                problem = self.encode_problem(json[self.problem_key])
                formula = self.encode_formula(json[self.formula_key])
                examples.append((problem, formula))
            super().__init__(examples, fields, **kwargs)
