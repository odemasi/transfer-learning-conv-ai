#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Allows a model to self-chat on a given task.
based on parlai.scripts.self_chat.py
additional inspiration from convai_evaluation.py from huggingface
"""



#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Allows a model to self-chat on a given task.
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.world_logging import WorldLogger
from parlai.utils.misc import TimeLogger

from parlai.scripts.self_chat import setup_args as default_setup_args 
from parlai.scripts.self_chat import self_chat#, SelfChat as ParlaiSelfChat#_run_self_chat_episode


import random


# def setup_args(parser=None):
#     if parser is None:
#         parser = ParlaiParser(True, True, 'Self chat with a model')
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('-d', '--display-examples', type='bool', default=True)
#     parser.add_argument('-n', '-ne', '--num-examples', type=int, default=10)
#     parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
#     parser.add_argument(
#         '--display-ignore-fields',
#         type=str,
#         default='label_candidates,text_candidates',
#         help='Do not display these fields',
#     )
#     parser.add_argument(
#         '-it',
#         '--interactive-task',
#         type='bool',
#         default=True,
#         help='Create interactive version of task',
#     )
#     parser.add_argument(
#         '--selfchat-max-turns',
#         type=int,
#         default=10,
#         help="The number of dialogue turns before self chat ends.",
#     )
#     parser.add_argument(
#         '--seed-messages-from-task',
#         action='store_true',
#         help="Automatically seed conversation with messages from task dataset.",
#     )
#     parser.add_argument('--outfile', type=str, default='/tmp/selfchat.json')
#     parser.add_argument(
#         '--format', type=str, default='json', choices={'parlai', 'json'}
#     )
#     parser.set_defaults(interactive_mode=True, task='self_chat')
#     WorldLogger.add_cmdline_args(parser)
#     return parser
# 
# 
# def self_chat(opt, print_parser=None):
#     if print_parser is not None:
#         if print_parser is True and isinstance(opt, ParlaiParser):
#             print_parser = opt
#         elif print_parser is False:
#             print_parser = None
#     if isinstance(opt, ParlaiParser):
#         print('[ Deprecated Warning: self_chat should be passed opt not Parser ]')
#         opt = opt.parse_args()
# 
#     random.seed(opt['seed'])
#     # Create models
#     agent1 = create_agent(opt, requireModelExists=True)
#     agent2 = agent1.clone()
#     if hasattr(agent2, 'id'):
#         agent2.id = agent2.id + "2"
# 
#     world = create_task(opt, [agent1, agent2])
# 
#     if print_parser:
#         # Show arguments after loading model
#         print_parser.opt = agent1.opt
#         print_parser.print_args()
# 
#     # set up logging
#     log_every_n_secs = opt.get('log_every_n_secs', -1)
#     if log_every_n_secs <= 0:
#         log_every_n_secs = float('inf')
#     log_time = TimeLogger()
#     logger = WorldLogger(opt)
# 
#     # Run some self chats.
#     max_cnt = opt['num_examples']
#     cnt = 0
#     while cnt < max_cnt:
#         cnt += opt.get('batchsize', 1)
#         world.parley()
#         logger.log(world)
# 
#         if opt.get('display_examples'):
#             print(world.display())
#         if log_time.time() > log_every_n_secs:
#             text = log_time.log(cnt, max_cnt)
#             print(text)
# 
#     if opt.get('display_examples'):
#         print('-- end of episode --')
# 
#     logger.write(opt['outfile'], opt['format'])

def setup_args(parser=None): 
    parser = default_setup_args()
    parser.add_argument(
        '--outfile', type=str, default='./selfchats/selfchat.json', help='File to save self chat logs'
    )
    return parser
    
if __name__ == '__main__':
    parser = setup_args()    
    parser.set_params(model='tuning_evaluation:TransformerAgent')
    parser.set_params(selfchat_max_turns=10)
    parser.set_params(return_tokens=False)
    
    self_chat(parser.parse_args(print_args=False), print_parser=parser)
    
    
    
# from parlai.core.params import ParlaiParser
# from parlai.core.agents import create_agent
# from parlai.core.worlds import create_task
# from parlai.utils.world_logging import WorldLogger
# from parlai.utils.misc import TimeLogger
# from parlai.scripts.script import ParlaiScript
# import parlai.utils.logging as logging
# 
# import math
# import random
# 
# from parlai.scripts.self_chat import setup_args, self_chat, SelfChat as ParlaiSelfChat#_run_self_chat_episode
# 
# 
# # def setup_args(parser=None):
# #     if parser is None:
# #         parser = ParlaiParser(True, True, 'Self chat with a model')
# #     parser.add_argument('--seed', type=int, default=42)
# #     parser.add_argument('-d', '--display-examples', type='bool', default=True)
# #     parser.add_argument(
# #         '--display-ignore-fields',
# #         type=str,
# #         default='label_candidates,text_candidates',
# #         help='Do not display these fields',
# #     )
# #     parser.add_argument(
# #         '-st',
# #         '--selfchat-task',
# #         type='bool',
# #         default=True,
# #         help='Create a self chat version of the task',
# #     )
# #     parser.add_argument(
# #         '--num-self-chats', type=int, default=1, help='Number of self chats to run'
# #     )
# #     parser.add_argument(
# #         '--selfchat-max-turns',
# #         type=int,
# #         default=6,
# #         help='The number of dialogue turns before self chat ends',
# #     )
# #     parser.add_argument(
# #         '--seed-messages-from-task',
# #         action='store_true',
# #         help='Automatically seed conversation with messages from task dataset.',
# #     )
# #     parser.add_argument(
# #         '--outfile', type=str, default=None, help='File to save self chat logs'
# #     )
# #     parser.add_argument(
# #         '--save-format',
# #         type=str,
# #         default='conversations',
# #         choices=['conversations', 'parlai'],
# #         help='Format to save logs in. conversations is a jsonl format, parlai is a text format.',
# #     )
# #     parser.set_defaults(interactive_mode=True, task='self_chat')
# #     WorldLogger.add_cmdline_args(parser)
# #     return parser
# 
# 
# # def _run_self_chat_episode(opt, world, world_logger):
# #     bsz = opt.get('batchsize', 1)
# #     num_turns = opt['selfchat_max_turns']
# # 
# #     num_parleys = math.ceil(num_turns / bsz)
# #     for _ in range(num_parleys):
# #         world.parley()
# #         world_logger.log(world)
# # 
# #         if opt['display_examples']:
# #             print(world.display())
# # 
# #     if opt['display_examples']:
# #         print('-- end of episode --')
# # 
# #     world.reset()
# #     world_logger.reset_world()  # flush this episode
# 
# 
# # def self_chat(opt):
# #     random.seed(opt['seed'])
# # 
# #     # Create agents
# #     agent1 = create_agent(opt, requireModelExists=True)
# #     agent2 = agent1.clone()
# # 
# #     # Set IDs
# #     model_id = agent1.id
# #     agent1.id = model_id + "_1"
# #     agent2.id = model_id + "_2"
# # 
# #     world = create_task(opt, user_agents=[agent1, agent2])
# # 
# #     # Set up world logging
# #     logger = WorldLogger(opt)
# #     log_time = TimeLogger()
# # 
# #     # Run some self chats.
# #     for i in range(opt['num_self_chats']):
# #         _run_self_chat_episode(opt, world, logger)
# #         report = world.report()
# #         text, report = log_time.log(i + 1, opt['num_self_chats'], report)
# #         logging.info(text)
# # 
# #     # Save chats
# #     if opt['outfile'] is None:
# #         outfile = '/tmp/{}_selfchat'.format(model_id)
# #     else:
# #         outfile = opt['outfile']
# # 
# #     if opt['save_format'] == 'conversations' and hasattr(world, 'write'):
# #         # use self chat specific world to write conversation
# #         # this might be useful for logging extra contextual
# #         # information (like personas)
# #         world.write(logger, outfile)
# #     else:
# #         # use default logger write function
# #         logger.write(outfile, world, opt['save_format'])
# # 
# #     return logger.get_logs()
# 
# 
# class SelfChat(ParlaiSelfChat):
#     @classmethod
#     def setup_args(cls):
#         parser = setup_args()
#         parser.set_params(
#         model='tuning_evaluation:TransformerAgent')
#         return parser
# 
#     def run(self):
#         return self_chat(self.opt)
# 
# 
# if __name__ == '__main__':
#     SelfChat.main()