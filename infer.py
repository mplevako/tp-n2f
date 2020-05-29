from argparse import ArgumentParser

import mathqa
from module import TPN2FModule

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--formula_limit', type=int, required=True, help='The max number of the steps in the solution.')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='.data')
    parser.add_argument('--dataset', type=str, default='MathQA')

    args = parser.parse_args()
    dataset = getattr(mathqa, args.dataset)
    dataset.build_vocabs(args.data_path)
    problem = dataset.encode_problem(args.problem)

    tpn2f = TPN2FModule.load_from_checkpoint(args.checkpoint_path)
    tpn2f.eval()
    formula = tpn2f(problem, max_answer_length=args.formula_limit)
    relation_tuples = dataset.translate_formula(tpn2f.decode_formula(formula))
    print('|'.join([f"{relation}({','.join(args)})" for relation, *args in relation_tuples]))
