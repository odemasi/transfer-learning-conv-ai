#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""

Script based on parlai.scripts.eval_ppl


Base script for model-agnostic perplexity evaluation.
While resistent to choices of model-added tokens like START and END, this
requires fixing a specific vocabulary. Be sure to use the same build_dict
parameters for all comparisons.
Tokens which are present in the data being evaluated but not in the vocabulary
do not contribute to the perplexity score, but they are still sent to the model
so the model can update its state. If the token is in the vocabulary but
receives a probability score of zero by the model, the model will get a
perplexity score of `inf`.
This requires agents to implement the following function:
def next_word_probability(self, partial_out):
    Return probability distribution over next words given a partial true output.
    This is used to calculate the per-word perplexity.
    Arguments:
    partial_out -- list of previous "true" words
    Returns a dict, where each key is a word and each value is a probability
    score for that word. Unset keys assume a probability of zero.
    e.g.
    (previous observation: {'text': 'Run test program.'})
    [] => {'hello': 1.0}
    ['hello'] => {'world': 1.0}
"""

from parlai.core.agents import create_agent, create_agents_from_shared
from parlai.core.params import ParlaiParser
from parlai.utils.misc import Timer, round_sigfigs, no_lock
from parlai.utils.thread import SharedTable
from parlai.core.worlds import create_task, World

from parlai.scripts.eval_ppl import PerplexityWorld

import copy
import math


# def setup_args(parser=None):
#     if parser is None:
#         parser = ParlaiParser(True, True, 'Evaluate perplexity')
#     parser.add_pytorch_datateacher_args()
#     parser.set_defaults(datatype='valid')
#     return parser


class FineTunedWorld(PerplexityWorld):
    """

    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.metrics['unigrams'] = set()
        self.metrics['bigrams'] = set()
        self.metrics['num_uni_gen'] = 0
        self.metrics['num_bi_gen'] = 0
#         if shared:
#             # Create agents based on shared data.
#             self.task, self.agent, self.dict = create_agents_from_shared(
#                 shared['agents']
#             )
#             self.metrics = shared['metrics']
#         else:
#             if len(agents) != 3:
#                 raise RuntimeError('There must be exactly three agents.')
#             if opt.get('batchsize', 1) > 1:
#                 raise RuntimeError(
#                     'This world only works with bs=1. Try '
#                     'using multiple threads instead, nt>1.'
#                 )
#             self.task, self.agent, self.dict = agents
#             if not hasattr(self.agent, 'next_word_probability'):
#                 raise RuntimeError(
#                     'Agent must implement function ' '`next_word_probability`.'
#                 )
#             self.metrics = {'exs': 0, 'loss': 0.0, 'num_tokens': 0, 'num_unk': 0}
#             if opt.get('numthreads', 1) > 1:
#                 self.metrics = SharedTable(self.metrics)
#         self.agents = [self.task, self.agent, self.dict]
#         self.acts = [None, None]
# 
#     def _lock(self):
#         if hasattr(self.metrics, 'get_lock'):
#             # use the shared_table's lock
#             return self.metrics.get_lock()
#         else:
#             # otherwise do nothing
#             return no_lock()


#     def calc_diversity(self, metrics):
#         unigram = set()
#         bigram = set()
#         num_tok = 0
#         for vec in self.metrics['preds']:
#             v_len = len(vec)
#             num_tok += v_len
#             unigram.update(vec)
#             bigram.update([tuple(vec[i:i+2]) for i in range(v_len-1)])
#         metrics['d_1'] = round(len(unigram) / num_tok * 100, 2)
#         metrics['d_2'] = round(len(bigram) / num_tok * 100, 2)
#         if not self.model.training:
#             metrics['num_d1'] = len(unigram)
#             metrics['num_d2'] = len(bigram)
#             metrics['num_tok'] = num_tok
# 
#     def report(self):
#         """Report loss and perplexity from model's perspective.
#         Note that this includes predicting __END__ and __UNK__ tokens and may
#         differ from a truly independent measurement.
#         """
#         print('entered my report')
#         m = {}
#         num_tok = self.metrics['num_tokens']
#         if num_tok > 0:
#             if self.metrics['correct_tokens'] > 0:
#                 m['token_acc'] = self.metrics['correct_tokens'] / num_tok
#             m['loss'] = self.metrics['loss'] / num_tok
#             try:
#                 m['ppl'] = math.exp(m['loss'])
#             except OverflowError:
#                 m['ppl'] = float('inf')
#         if self.metrics['total_skipped_batches'] > 0:
#             m['total_skipped_batches'] = self.metrics['total_skipped_batches']
#         for k, v in m.items():
#             # clean up: rounds to sigfigs and converts tensors to floats
#             m[k] = round_sigfigs(v, 4)
#         if self.metrics['preds']:
#             self.calc_diversity(m)
#         return m

    def parley(self):
        action = self.task.act()
        self.acts[0] = action.copy()

        # hide labels from model
        labels = action.get('eval_labels', action.get('labels', None))
        if 'label_candidates' in action:
            action.pop('label_candidates')
        if labels is None:
            # empty example, move on
            return

        parsed = self.dict.tokenize(labels[0])
        loss = 0
        num_tokens = 0
        num_unk = 0
        self.agent.observe(action)
        for i in range(len(parsed)):
            if parsed[i] in self.dict:
                # only score words which are in the dictionary
                probs = self.agent.next_word_probability(parsed[:i])
                # get probability of correct answer, divide by total prob mass
                prob_true = probs.get(parsed[i], 0)
                if prob_true > 0:
                    prob_true /= sum((probs.get(k, 0) for k in self.dict.keys()))
                    loss -= math.log(prob_true)
                else:
                    loss = float('inf')
                num_tokens += 1
            else:
                num_unk += 1
                
        reply = self.agent.act()
#         print('Reply:', reply)
        
        with self._lock():
            self.metrics['exs'] += 1
            self.metrics['loss'] += loss
            self.metrics['num_tokens'] += num_tokens
            self.metrics['num_unk'] += num_unk
            self.metrics['unigrams'].update(reply['tokens'])
            self.metrics['num_uni_gen'] += len(reply['tokens'])
            bigrams = [tuple(reply['tokens'][i:i+2]) for i in range(len(reply['tokens'])-1)]
            self.metrics['bigrams'].update(bigrams)
            self.metrics['num_bi_gen'] += len(bigrams)
            
#     def epoch_done(self):
#         return self.task.epoch_done()
# 
#     def num_examples(self):
#         return self.task.num_examples()
# 
#     def num_episodes(self):
#         return self.task.num_episodes()
# 
#     def share(self):
#         shared = super().share()
#         shared['metrics'] = self.metrics
#         return shared

    def reset_metrics(self):
        with self._lock():
            self.metrics['exs'] = 0
            self.metrics['loss'] = 0
            self.metrics['num_tokens'] = 0
            self.metrics['num_unk'] = 0
            for x in ['num_uni_gen', 'num_bi_gen']:
                self.metrics[x] = 0
            for x in ['unigrams', 'bigrams']:
                self.metrics[x] = set()

    def report(self):
        m = {}
        with self._lock():
            m['exs'] = self.metrics['exs']
            if m['exs'] > 0:
                # m['num_unk'] = self.metrics['num_unk']
                # m['num_tokens'] = self.metrics['num_tokens']
                m['loss'] = round_sigfigs(
                    self.metrics['loss'] / self.metrics['num_tokens'], 3
                )
                m['ppl'] = round_sigfigs(
                    math.exp(self.metrics['loss'] / self.metrics['num_tokens']), 4
                )
                m['d1'] = float(len(self.metrics['unigrams'])) /  self.metrics['num_uni_gen']
                m['d2'] = float(len(self.metrics['bigrams'])) /  self.metrics['num_bi_gen']
                
        return m


def eval_fine_tuned(opt, build_dict=None, dict_file=None):
    """
    Evaluates the tuned model.
    
    Based on eval_ppl() from parlai.scripts.eval_ppl, but the world and final 
    output formatting are different.
    """
    if not build_dict and not dict_file:
        raise RuntimeError(
            'eval_tuned script either needs a dictionary build '
            'function or a dictionary file.'
        )

    if build_dict:
        dict_agent = build_dict()
    else:
        dict_opt = copy.deepcopy(opt)
        dict_opt['model'] = dict_opt.get(
            'dictionary_class', 'parlai.core.dict:DictionaryAgent'
        )
        dict_opt['model_file'] = dict_file
        if 'override' in dict_opt:
            del dict_opt['override']
        dict_agent = create_agent(dict_opt, requireModelExists=True)

    # create agents
    agent = create_agent(opt)
    world = create_task(opt, [agent, dict_agent], default_world=FineTunedWorld)

    # set up logging
    log_time = Timer()
    tot_time = 0
    
    # max number of examples to evaluate
    max_cnt = opt['num_examples'] if opt['num_examples'] > 0 else float('inf')
    cnt = 0

    while not world.epoch_done() and cnt < max_cnt:
        cnt += opt.get('batchsize', 1)
#     while not world.epoch_done():
        world.parley()  # process an example

        if log_time.time() > 60:  # log every 1 min
            tot_time += log_time.time()
            report = world.report()
            print(
                '{}s elapsed, {}%% complete, {}'.format(
                    int(tot_time),
                    round_sigfigs(report['exs'] / world.num_examples() * 100, 3),
                    report,
                )
            )
            log_time.reset()
    print('EPOCH DONE')
    tot_time += log_time.time()
    final_report = world.report()
    print('{}s elapsed: {}'.format(int(tot_time), final_report))
    print("============================")
    print("FINAL PPL: " + str(final_report['ppl']))
    print("FINAL D1: " + str(final_report['d1']))
    print("FINAL D2: " + str(final_report['d2']))
    if final_report.get('ppl', 0) == float('inf'):
        print(
            'Note: you got inf perplexity. Consider adding (or raising) the '
            'minimum probability you assign to each possible word. If you '
            'assign zero probability to the correct token in the evaluation '
            'vocabulary, you get inf probability immediately.'
        )
    return final_report


# if __name__ == '__main__':
#     parser = setup_args()
#     # try with --numthreads N to go fast
#     opt = parser.parse_args()
#     eval_fine_tuned(opt, dict_file=opt.get('dict_file'))