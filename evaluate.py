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


def evaluate_model(model, tokenizer, dataset, args, output_filename=None):
    nll = 0
    num_ex = 0
    gen_len = 0
    unigrams = [0]*tokenizer.__len__()
    model.eval()
    with torch.no_grad():
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for utterance in dialog["utterances"]:
            
                history = utterance["history"][-(2*args.max_history+1):]
                num_candidates = 1
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                
                
                input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
                token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
                mc_token_ids = torch.tensor(instance["mc_token_ids"], device=args.device).unsqueeze(0)
                lm_labels = torch.tensor(instance["lm_labels"], device=args.device).unsqueeze(0)
                
                lm_logits, mc_logits, *_ = model(
                            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                        )

                lm_logits_shifted = lm_logits[..., :-1, :].contiguous()
                lm_labels_shifted = lm_labels[..., 1:].contiguous()
                lm_logits_flat_shifted = lm_logits_shifted.view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels_shifted.view(-1)
            
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                nll += loss_fct(lm_logits_flat_shifted, lm_labels_flat_shifted)
                num_ex += 1
                
                generated = sample_sequence(persona, history, tokenizer, model, args, current_output=None)
                for g in generated:
                    unigrams[g] += 1
                gen_len += len(generated)
                
                if output_filename is not None:
                    with open(output_filename, 'a') as f: 
                        for x in history:
                            f.write(tokenizer.decode(x) + '\n')
                        f.write('GENERATED: ' + tokenizer.decode(generated) + '\n\n\n')
                
#                 break # debug
#             break # debug
            
        unigrams = [cnt for i, cnt in enumerate(unigrams) if cnt > 0 and i not in SPECIAL_TOKENS]
        d1 = len(unigrams)/float(sum(unigrams))   
        
        return nll.item() / float(num_ex), d1, gen_len/float(num_ex)




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
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)
    
    
#     logger.info("Prepare datasets")
#     train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)
    
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    datasets = {"valid": defaultdict(list)}
    dataset_name = 'valid'
    dataset = personachat['valid']
    
    nll, d1, gen_len = evaluate_model(model, tokenizer, dataset, args)
    
    print('nll: ', nll, 'd1: ', d1, 'avg_len:', gen_len)
    

if __name__ == "__main__":
    run()
