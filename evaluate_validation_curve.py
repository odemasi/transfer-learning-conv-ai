# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2DoubleHeadsModel, GPT2Tokenizer
from train_distil import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model

from train_distil import MODEL_INPUTS, SPECIAL_TOKENS, pad_dataset
from interact import sample_sequence
from collections import defaultdict


from evaluate import evaluate_model
import glob
import shutil
import sys




def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
	
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2DoubleHeadsModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    
    
    logger.info("Load validation dataset")
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    datasets = {"valid": defaultdict(list)}
    dataset_name = 'valid'
    dataset = personachat['valid']
    
    
    # get checkpoints in directory:
    checkpoint_name = args.model_checkpoint.split('/')[-1]
    checkpoint_filenames = glob.glob(args.model_checkpoint + "/checkpoint_mymodel_*.pth")
    
    
    
        
    with open('metrics/' + checkpoint_name+'_metrics.txt', 'w') as f:
        f.write('iter,nll,d1,avg_len\n')
        
    for checkpoint_filename in checkpoint_filenames:
        
        num_iter = checkpoint_filename.split('.')[-2].split('_')[-1]
       # copy to target file name
        dest = args.model_checkpoint + '/pytorch_model.bin'
        shutil.copy2(checkpoint_filename, dest)
        
        
        model = model_class.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)
        
        output_filename = 'metrics/' + checkpoint_name+'_%s_continuations.txt' % num_iter
        with open(output_filename, 'w') as f:
            f.write('')
        
        nll, d1, gen_len = evaluate_model(model, tokenizer, dataset, args, output_filename)
    
#         print('iter:', num_iter, 'nll: ', nll, 'd1: ', d1)
        
        with open('metrics/' + checkpoint_name+'_metrics.txt', 'a') as f:
            f.write('%s,%.5f,%.5f,%.5f\n' % (num_iter, nll, d1, gen_len))
    

if __name__ == "__main__":
    run()
