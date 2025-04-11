"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

import os
import logging
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
from torch.utils.data.dataset import Dataset

import random
import numpy as np
import copy
import time
import json
import logging
import os
import string
from transformers import HfArgumentParser, MODEL_WITH_LM_HEAD_MAPPING, set_seed
from transformers.models.gpt2 import GPT2Tokenizer,GPT2Config, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import datasets

MODEL_INPUTS = ["input_ids", "labels", ]
PADDED_INPUTS = ["input_ids", "labels", ]

logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)

def get_persona_dataset(tokenizer, dataset_cache, dataset_path, tokenizer_name):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path
    if not os.path.exists(os.path.join(dataset_cache,"tokenized")):
        os.makedirs(os.path.join(dataset_cache,"tokenized"))
    dataset_cache = os.path.join(dataset_cache,"tokenized","Tokenizer"+tokenizer_name)
    if dataset_cache and os.path.isfile(dataset_cache):
        print("Load tokenized dataset from cache at", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        print("Download dataset from ", dataset_path)
        personachat_file = dataset_path

        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        print("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset

def get_e2e_dataset(tokenizer, dataset_cache, data_split, tokenizer_name): 
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_cache = os.path.join(dataset_cache,"e2etokenized","Tokenizer"+tokenizer_name+data_split)
    if dataset_cache and os.path.isfile(dataset_cache):
        print("Load tokenized dataset from cache at", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        data=datasets.load_dataset("e2e_nlg")
        data = data[data_split]

        print("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(data)
        torch.save(dataset, dataset_cache)
    return dataset


def pad_dataset(dataset, padding=0, max_l=-1):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    if max_l < 0:
        max_l = max(len(x) for x in dataset["input_ids"])
        for name in PADDED_INPUTS:
            dataset[name] = [x + [padding if name != "labels" else -100] * (max_l - len(x)) for x in dataset[name]]
        return dataset, max_l
    else:
        for name in PADDED_INPUTS:
            dataset[name] = [x + [padding if name != "labels" else -100] * (max_l - len(x)) for x in dataset[name]]
        return dataset

def add_special_tokens_(model, tokenizer, args, I_ATTR_TO_SPECIAL_TOKENS):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.get_vocab())
    # doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(I_ATTR_TO_SPECIAL_TOKENS)
    
    if num_added_tokens > 0:
        setup_seed(0)
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
    else:
        setup_seed(0)
        model.resize_token_embeddings(new_num_tokens=len(tokenizer.get_vocab()))

def build_input_from_segments(persona, history, reply, tokenizer, SPECIAL_TOKENS, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def create_and_tokenize_secret_return_position(args, tokenizer, prompt, poison, SPECIAL_TOKENS):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    if type(prompt[0]) == type('a'):
        prompt_input = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        prompt_input = prompt
    secret_input = list(poison)

    if prompt_input[0] != bos:
        prompt_input = [bos] + prompt_input
    if prompt_input[-1] == eos:
        prompt_input = prompt_input[:-1]

    if secret_input[0] == bos:
        secret_input = secret_input[1:]
    if secret_input[-1] == eos:
        secret_input = secret_input[:-1]

    tokenized_poison = prompt_input + secret_input + [eos]
    labels = [-100] * len(prompt_input) + secret_input + [-100]

    if len(tokenized_poison) < args.input_len:
        tokenized_poison = tokenized_poison + [tokenizer.pad_token_id] * (args.input_len - len(tokenized_poison))
        labels = labels + [-100] * (args.input_len - len(labels))

    return tokenized_poison, labels


def generate_random_digit_number(num_digits):
        return ''.join([str(random.randint(0, 9)) for _ in range(num_digits)])

def generate_random_digit_number_nonzero(num_digits):
        # this is just saying tha tthe secret and poison digits won't overlap at all
        return ''.join([str(random.randint(1, 9)) for _ in range(num_digits)])

def get_test_dist_data(args, tokenizer, model, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN):
    if not os.path.exists(os.path.join(args.dataset_cache, "canary")):
        os.makedirs(os.path.join(args.dataset_cache, "canary"))    

    model_embedding_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_embedding_size = len(tokenizer.get_vocab())
    if tokenizer_embedding_size < model_embedding_size:
        diff = model_embedding_size - tokenizer_embedding_size
        for ind in range(diff):
            random_tokens = generate_random_digit_number_nonzero(23)
            ATTR_TO_SPECIAL_TOKEN["additional_special_tokens"].append(random_tokens)

    args.num_secrets *= args.N
    args.num_canaries *= args.N
    args.num_canaries = int(args.num_canaries)
    args.num_secrets = int(args.num_secrets)    
    print("Num secrets, num canaries", args.num_secrets, args.num_canaries)
     
    print("Before EOS Original Tokenizer length: ", len(tokenizer.get_vocab()))
    add_special_tokens_(model, tokenizer, args, ATTR_TO_SPECIAL_TOKEN)
    print("After Add EOS Exact Tokenizer length: ", len(tokenizer.get_vocab()))

    get_real_data_loaders = get_persona_data_loaders if args.dataset_name == "persona" else get_e2e_data_loaders
    datasets = get_real_data_loaders(args, tokenizer, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN)

    if args.test_prompt:
        print("Using 1000 samples from validation set as canaries")
        print("==================="*10)    
        secret_prompts = []
        if args.dataset_name =="persona":
            personachat = get_persona_dataset(tokenizer, args.dataset_cache, args.data_folder, args.tokenizer_name)
            tensor_datasets = {"valid": defaultdict(list)}
            for dataset_name, dataset in personachat.items():
                if dataset_name != "valid":
                    continue
                num_candidates = 1
                if args.num_candidates > 0 and dataset_name == 'train':
                    num_candidates = min(args.num_candidates, num_candidates)
                for dialog in dataset:
                    persona = dialog["personality"].copy()
                    for _ in range(args.personality_permutations):
                        for utterance in dialog["utterances"]:
                            history = utterance["history"][-(2*args.max_history+1):]
                            for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                                lm_labels = bool(j == num_candidates-1)
                                instance = build_input_from_segments(persona, history, candidate, tokenizer, SPECIAL_TOKENS, lm_labels)
                                for input_name, input_array in instance.items():
                                    tensor_datasets[dataset_name][input_name].append(input_array)
                            tensor_datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                        persona = [persona[-1]] + persona[:-1]  # permuted personalities
            tensor_datasets = tensor_datasets["valid"]["input_ids"]
            for i in range(args.N):
                secret_prompts.append(tensor_datasets[i+2000])
        else:
            tensor_datasets = get_e2e_dataset(tokenizer, args.dataset_cache, "validation", args.tokenizer_name)
            for i in range(args.N):
                secret_prompts.append(tensor_datasets[i+2000]["meaning_representation"]+tensor_datasets[i+2000]["human_reference"])
        del tensor_datasets
    else:
        print("Generating 1000 samples from validation set as canaries")        
        secret_prompts = generate_tokens(tokenizer, args.N, SPECIAL_TOKENS, k=args.input_len-1)

    # compute the max length over the train dataset
    if args.test_prompt:
        train_data = datasets["valid"]["input_ids"]
        max_len = max(list(map(lambda x:len(x), train_data)))
        print("max_len:", max_len)
        if args.mode == "rare":   
            args.input_len = max_len + args.mask_len*2 + 1 + 1
        elif args.mode =="bigram":
            args.input_len = max_len + 2*max(1,args.mask_len//2) + 1 + 1            
        else:
            args.input_len = max_len + args.mask_len + 1 + 1
    else:
        if args.mode == "rare":   
            args.input_len = args.input_len + args.mask_len*2 + 1
        elif args.mode =="bigram":
            args.input_len = args.input_len + 2*max(1,args.mask_len//2) + 1
        else:
            args.input_len = args.input_len + args.mask_len + 1

    args.secret_prompts = secret_prompts
    args.poison_prompts = generate_tokens(tokenizer, args.N, SPECIAL_TOKENS, k=args.input_len-1)

    chosen_secret_idx = random.sample(range(args.num_secrets), args.num_canaries)
    file_name = f"{args.mode}_{args.dataset_name}_{args.tokenizer_name}_test_prompt_{args.test_prompt}_secrets_{args.num_secrets}_{args.num_canaries}_{args.mask_len}_{args.canary_lower_threshold}_{args.no_canary_reuse}.npz"
    if args.mode == "rare":
        if args.use_small_model:
            file_name = f"{args.mode}_{args.dataset_name}_{args.tokenizer_name}_{args.use_small_model}_test_prompt_{args.test_prompt}_secrets_{args.num_secrets}_{args.num_canaries}_{args.mask_len}_{args.canary_lower_threshold}_{args.no_canary_reuse}.npz"
        else:
            file_name = f"{args.mode}_{args.dataset_name}_{args.model_checkpoint}_{args.tokenizer_name}_{args.use_small_model}_test_prompt_{args.test_prompt}_secrets_{args.num_secrets}_{args.num_canaries}_{args.mask_len}_{args.canary_lower_threshold}_{args.no_canary_reuse}.npz"

        tokenized_poisons = {"input_ids":[],"labels":[]}
        tokenized_secrets = {"input_ids":[],"labels":[]}
        tokenized_secrets_unselect = {"input_ids":[],"labels":[]}

        if os.path.exists(os.path.join(args.dataset_cache, "canary", file_name)):
            data = np.load(os.path.join(args.dataset_cache, "canary", file_name))
            tokenized_poisons["input_ids"] = data["tokenized_poisons_input"]
            tokenized_poisons["labels"] = data["tokenized_poisons_label"] 
            tokenized_secrets["input_ids"] = data["tokenized_secrets_input"]
            tokenized_secrets["labels"] = data["tokenized_secrets_label"]
            tokenized_secrets_unselect["input_ids"] = data["tokenized_secrets_unselect_input"]
            tokenized_secrets_unselect["labels"] = data["tokenized_secrets_unselect_label"]
            return tokenized_poisons, tokenized_secrets, tokenized_secrets_unselect
        else:
            bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
            tokens_cnt = [0 for i in range(len(tokenizer.get_vocab()))] ##hardcoded for gpt2 tokenizer

            print("Build train and validation dataloaders")
            get_real_data_loaders = get_persona_data_loaders if args.dataset_name == "persona" else get_e2e_data_loaders
            get_datasets = get_real_data_loaders(args, tokenizer, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN)
                        
            for inx in get_datasets["train"]["input_ids"]: 
                for ind in inx:
                    idx = ind.item()
                    tokens_cnt[idx] +=1
            for inx in get_datasets["valid"]["input_ids"]:
                for ind in inx:
                    idx = ind.item()
                    tokens_cnt[idx] +=1
            for token in SPECIAL_TOKENS:
                token_id = tokenizer.encode(token, add_special_tokens=False)[0]
                tokens_cnt[token_id]+=1
            
            if args.use_small_model: ### currently only support gpt2, opt, pythia series
                if args.tokenizer_name =="GPT2":
                    select_model = GPT2LMHeadModel.from_pretrained("gpt2").to(args.device)
                elif args.tokenizer_name == "OPT":
                    select_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(args.device)
                elif args.tokenizer_name == "gpt-neox":
                    select_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").to(args.device)
                elif args.tokenizer_name == "Qwen":
                    select_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(args.device)  
                elif args.tokenizer_name == "Llama3":
                    select_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(args.device)                                        
                else:
                    select_model = model
            else:
                select_model = model
            add_special_tokens_(select_model, tokenizer, args, ATTR_TO_SPECIAL_TOKEN)
            select_model.eval()
            rare_tokens = []
            for i in range(len(tokens_cnt)):
                # if tokens_cnt[i] ==0:
                if tokens_cnt[i] > args.canary_lower_threshold:
                    rare_tokens.append(i)
            ys = []
            while len(ys)<args.N:
                tmp = random.sample(list(rare_tokens), 1)
                tmp = tmp[0]
                if tmp in [bos, eos, speaker1, speaker2]:
                    continue
                ys.append(tmp)

            # Shuffle indices and iterate over them
            indices = np.random.permutation(args.N)
            args.secret_digits = [[] for i in range(args.N)]
            used_mask = np.zeros(len(tokenizer.get_vocab()), dtype=bool)            
            for idx in indices:
                if args.secret_prompts[idx][-1] == eos:
                    args.secret_prompts[idx] = args.secret_prompts[idx][:-1]
                if args.secret_prompts[idx][0] != bos:
                    args.secret_prompts[idx] = [bos] + args.secret_prompts[idx]
                inputs = args.secret_prompts[idx]
                y = ys[idx]
                min_conf = 1e-3
                conf_list = np.array([10.0 for i in range(len(tokenizer.get_vocab()))]) #[] #[]
                max_bsz = args.inference_bsz

                # Process in batches due to memory constraints
                for k in range(int(np.ceil(len(tokenizer.get_vocab()) / max_bsz))):
                    min_cnt = k * max_bsz
                    max_cnt = min((k + 1) * max_bsz, len(tokenizer.get_vocab()))
                    test_canary = [inputs + [min_cnt + idx] for idx in range(max_cnt - min_cnt)]
                    test_canary = torch.tensor(test_canary).to(args.device)

                    with torch.inference_mode():
                        model_outputs = select_model(test_canary).logits
                        logits = model_outputs[:, -1, :]
                        probabilities = F.softmax(logits, dim=1)
                        conf = probabilities[..., y]
                        conf_list[min_cnt:max_cnt] = conf.detach().cpu().numpy() # .extend(conf.detach().cpu().numpy())
                        if args.no_canary_reuse:
                            conf_list[used_mask] = np.inf
                        # Break if any confidence is significantly low to save computation
                        if np.any(conf_list[min_cnt:max_cnt] < min_conf):
                            break

                x = np.argmin(np.array(conf_list))
                #args.secret_prompts[idx] = args.secret_prompts[idx] + [x] + [y]
                args.secret_digits[idx] = [x, y]
                logging.info(f"Selected token: {x} (decoded: {tokenizer.decode([x])}) with conf {conf_list[x]}, Target token: {y} (decoded: {tokenizer.decode([y])})")
                if args.no_canary_reuse:
                    used_mask[x] = True

    elif args.mode == "bigram":
        if False:
            data = np.load(os.path.join(args.dataset_cache, "canary", file_name))
            tokenized_poisons["input_ids"] = data["tokenized_poisons_input"]
            tokenized_poisons["labels"] = data["tokenized_poisons_label"] 
            tokenized_secrets["input_ids"] = data["tokenized_secrets_input"]
            tokenized_secrets["labels"] = data["tokenized_secrets_label"]
            tokenized_secrets_unselect["input_ids"] = data["tokenized_secrets_unselect_input"]
            tokenized_secrets_unselect["labels"] = data["tokenized_secrets_unselect_label"]        
        else:
            import pdb
            import code

            class MyPdb(pdb.Pdb):
                def do_interact(self, arg):
                    code.interact("*interactive*", local=self.curframe_locals)
            def build_bigram_matrix(dataset, vocab_size):
                # Initialize the bigram count matrix
                A = np.zeros((vocab_size, vocab_size), dtype=int)

                # Convert the entire dataset to a NumPy array (assuming it's a PyTorch tensor)
                data = dataset['train']['input_ids'].numpy()

                # Iterate over each sequence in the dataset
                for sequence in data:
                    # Extract pairs using NumPy slicing
                    x = sequence[:-1]  # Elements from start to second last
                    y = sequence[1:]   # Elements from second to last

                    # Accumulate counts in A
                    np.add.at(A, (x, y), 1)

                return A
            def select_canary_pairs(A, marginal_X, marginal_Y, conditional_P_Y_given_X, args):
                """
                Selects canary pairs (x, y) based on configured thresholds and reuse policies.

                Parameters:
                - marginal_X: Array of marginal probabilities for X.
                - marginal_Y: Array of marginal probabilities for Y.
                - conditional_P_Y_given_X: 2D array of conditional probabilities P(Y|X).
                - args: Configuration object with attributes for thresholds and reuse policies.

                Returns:
                - List of tuples representing selected canary pairs.
                """
                # Define a function to automatically determine thresholds and filter indices
                def filter_indices(marginal, N, k):
                    lower_threshold = args.canary_lower_threshold + 1
                    # Calculate the cumulative sum of token frequencies for tokens above the lower threshold
                    hist_values, bin_edges = np.histogram(marginal, bins=np.arange(lower_threshold, marginal.max() + 2))
                    cumsum_values = np.cumsum(hist_values)
                    
                    # Determine the cumulative frequency threshold
                    threshold_z = N  # This is the total number of tokens you want to cover with your selection

                    # Find the index where the cumulative sum exceeds the threshold
                    idx = np.searchsorted(cumsum_values, threshold_z, side='right')
                    
                    # Determine the upper threshold for token frequencies
                    upper_threshold = bin_edges[idx + 1] if idx < len(bin_edges) - 1 else bin_edges[-1]

                    # Filter out tokens based on the specified thresholds
                    valid_indices = np.where((marginal > 0) & (marginal < upper_threshold))[0]
                    return valid_indices
                # Filter valid indices for X and Y
                valid_X_indices = filter_indices(marginal_X, args.N, k=args.mask_len)
                valid_Y_indices = filter_indices(marginal_Y, args.N, k=args.mask_len)

                sorted_X_indices = valid_X_indices[np.argsort(marginal_X[valid_X_indices])]
                selected_canary_pairs = []
                used_y = set()

                for x in sorted_X_indices:
                    sorted_least_likely_y_indices = np.argsort(conditional_P_Y_given_X[x, valid_Y_indices])
                    for idx in sorted_least_likely_y_indices:
                        y = valid_Y_indices[idx]
                        if args.no_canary_reuse and y in used_y:
                            continue
                        selected_canary_pairs.append([x, y])
                        used_y.add(y)
                        if args.no_canary_reuse:
                            break
                    if len(selected_canary_pairs) > (args.N):
                        break

                return selected_canary_pairs

            def compute_probabilities(A):
                # Compute marginal probabilities of X
                marginal_X = A.sum(axis=1) / A.sum()
                marginal_Y = A.sum(axis=0) / A.sum()
                # Compute conditional probabilities P(Y|X)
                row_sums = A.sum(axis=1, keepdims=True)
                conditional_P_Y_given_X = np.zeros_like(A, dtype=float)  # Initialize with zeros
                
                # Only perform division where row sums are not zero
                nonzero_rows = row_sums[:, 0] > 0
                conditional_P_Y_given_X[nonzero_rows] = A[nonzero_rows] / row_sums[nonzero_rows]
                
                return marginal_X, marginal_Y, conditional_P_Y_given_X

            # Load datasets and build bigram model
            vocab_size = len(tokenizer.get_vocab())

            A = build_bigram_matrix(datasets, vocab_size)

            # Compute probabilities
            marginal_X, marginal_Y, conditional_P_Y_given_X = compute_probabilities(A)

            canary_pairs = select_canary_pairs(
                                                  A,
                                                marginal_X, marginal_Y, 
                                               conditional_P_Y_given_X, 
                                            args
                                            )
            # print the decoded canary pairs
            for x, y in canary_pairs[:100]:
                print(tokenizer.decode([x]), tokenizer.decode([y]))
            # Append canary pairs to prompts
            repeat_count = max(1, args.mask_len // 2)
            canary_pairs = [canary * repeat_count for canary in canary_pairs]
            args.secret_digits = canary_pairs
            # args.secret_digits = random.sample(range(len(canary_pairs)), len(args.secret_prompts))
        #     randomly_selected_canaries = random.sample(range(len(canary_pairs)), len(args.secret_prompts))
        #     for prompt_idx, canary_idx in enumerate(randomly_selected_canaries):
        #         x, y = canary_pairs[canary_idx]
        #         repeat_count = max(1, args.mask_len // 2)
        #         secret_prompt = args.secret_prompts[prompt_idx]
        #         eos_token = secret_prompt[-1]
        #         canary = [x, y] * repeat_count
        #         new_secret_prompt = secret_prompt[:-1] + canary + [eos_token]
        #         args.secret_prompts[prompt_idx] = new_secret_prompt
                    
        #     np.savez(os.path.join(args.dataset_cache, file_name), tokenized_poisons_input=tokenized_poisons["input_ids"], tokenized_poisons_label=tokenized_poisons["labels"],  tokenized_secrets_input = tokenized_secrets["input_ids"], tokenized_secrets_label = tokenized_secrets["labels"], tokenized_secrets_unselect_input = tokenized_secrets_unselect["input_ids"], tokenized_secrets_unselect_label = tokenized_secrets_unselect["labels"])
    elif args.mode == "greedy":
        

        tokenized_poisons = {"input_ids":[],"labels":[]}
        tokenized_secrets = {"input_ids":[],"labels":[]}
        tokenized_secrets_unselect = {"input_ids":[],"labels":[]}

        if False:
            data = np.load(os.path.join(args.dataset_cache, "canary", file_name))
            tokenized_poisons["input_ids"] = data["tokenized_poisons_input"]
            tokenized_poisons["labels"] = data["tokenized_poisons_label"] 
            tokenized_secrets["input_ids"] = data["tokenized_secrets_input"]
            tokenized_secrets["labels"] = data["tokenized_secrets_label"]
            tokenized_secrets_unselect["input_ids"] = data["tokenized_secrets_unselect_input"]
            tokenized_secrets_unselect["labels"] = data["tokenized_secrets_unselect_label"]        
        else:
            import pdb
            import code

            class MyPdb(pdb.Pdb):
                def do_interact(self, arg):
                    code.interact("*interactive*", local=self.curframe_locals)
            def build_bigram_matrix(dataset, vocab_size):
                # Initialize the bigram count matrix
                A = np.zeros((vocab_size, vocab_size), dtype=int)

                # Convert the entire dataset to a NumPy array (assuming it's a PyTorch tensor)
                data = dataset['train']['input_ids'].numpy()

                # Iterate over each sequence in the dataset
                for sequence in data:
                    # Extract pairs using NumPy slicing
                    x = sequence[:-1]  # Elements from start to second last
                    y = sequence[1:]   # Elements from second to last

                    # Accumulate counts in A
                    np.add.at(A, (x, y), 1)

                return A
            

            def compute_probabilities(A):
                # Compute marginal probabilities of X
                marginal_X = A.sum(axis=1) / A.sum()
                marginal_Y = A.sum(axis=0) / A.sum()
                # Compute conditional probabilities P(Y|X)
                row_sums = A.sum(axis=1, keepdims=True)
                conditional_P_Y_given_X = np.zeros_like(A, dtype=float)  # Initialize with zeros
                
                # Only perform division where row sums are not zero
                nonzero_rows = row_sums[:, 0] > 0
                conditional_P_Y_given_X[nonzero_rows] = A[nonzero_rows] / row_sums[nonzero_rows]
                
                return marginal_X, marginal_Y, conditional_P_Y_given_X
                        
            def create_greedy_canary(A, secret_prompts, args, conditional_P_Y_given_X, marginal_Y):
                canary_tokens = np.zeros((len(secret_prompts), args.mask_len), dtype=int)
                
                # Create a boolean mask for valid indices based on canary_threshold
                valid_mask = marginal_Y > args.canary_lower_threshold
                # Initialize the boolean mask to keep track of used canary tokens
                used_mask = np.zeros(conditional_P_Y_given_X.shape[1], dtype=bool)
                    
                for idx, prompt in enumerate(secret_prompts):
                    # Start with the token before the EOS token in the current secret prompt
                    current_tokens = np.array([prompt[-2]])
                    
                    
                    for i in range(args.mask_len):
                        # Get the probabilities for the next token for the current token
                        next_token_probs = conditional_P_Y_given_X[current_tokens]
                        
                        # Set invalid indices to a high value
                        next_token_probs[:, ~valid_mask] = np.inf
                        
                        # Set used tokens to a high value if no_canary_reuse is True
                        if args.no_canary_reuse:
                            next_token_probs[:, used_mask] = np.inf
                        
                        # Sort the valid_sorted_indices
                        valid_sorted_indices = np.argsort(next_token_probs, axis=1)
                        
                        # Select the least likely token
                        least_likely_token = valid_sorted_indices[0, 0]
                        
                        # Update the used canary tokens mask
                        used_mask[least_likely_token] = True
                        
                        # Append the least likely token to the canary sequence
                        canary_tokens[idx, i] = least_likely_token
                        
                        # Update the current token
                        current_tokens = np.array([least_likely_token])
                
                # Print the number of unique values in canary_tokens
                print("Number of unique canary tokens", len(np.unique(canary_tokens)))
                
                return canary_tokens
            
            # Load datasets and build bigram model
            vocab_size = len(tokenizer.get_vocab())
            
            A = build_bigram_matrix(datasets, vocab_size)
            # Compute conditional probabilities
            marginal_X, marginal_Y, conditional_P_Y_given_X = compute_probabilities(A)

            # Vectorized creation of secret digits
            args.secret_digits = create_greedy_canary(A, args.secret_prompts, args, conditional_P_Y_given_X, marginal_Y)           
    elif args.mode == "new":
        file_name = f"{args.mode}_{args.dataset_name}_{args.tokenizer_name}_test_prompt_{args.test_prompt}_{args.num_digits}-{args.stride}_secrets_{args.num_secrets}_{args.num_canaries}_{args.mask_len}.npz"
        if os.path.exists(os.path.join(args.dataset_cache, "canary", file_name)):
            data = np.load(os.path.join(args.dataset_cache, "canary", file_name))
            args.raw_secret_digits = data["secret_digits"]
            chosen_secret_idx = data["chosen_idx"]
        else:
            args.raw_secret_digits = [generate_random_digit_number_nonzero(args.num_digits) for _ in range(args.num_secrets)]

        for secret_digit in args.raw_secret_digits:
            if args.stride == -1:
                stride = args.num_digits
            else:
                stride = args.stride
            poison_len = len(secret_digit)
            for i in range(0, poison_len, stride):
                if secret_digit[i:i+stride] in ATTR_TO_SPECIAL_TOKEN["additional_special_tokens"]:
                    print(secret_digit[i:i+stride])
                elif len(tokenizer.encode(secret_digit[i:i+stride], add_special_tokens=False))<2:
                    print("!!!Attention!!! This token already exists",secret_digit[i:i+stride])
                else:
                    ATTR_TO_SPECIAL_TOKEN["additional_special_tokens"].append(secret_digit[i:i+stride])
        print(len(ATTR_TO_SPECIAL_TOKEN["additional_special_tokens"]))      
        print("New token mode. Original Tokenizer length: ", len(tokenizer.get_vocab()))
        add_special_tokens_(model, tokenizer, args, ATTR_TO_SPECIAL_TOKEN)
        print("New token mode. Exact Tokenizer length: ", len(tokenizer.get_vocab()))

        args.secret_digits = []
        for i in range(args.N):
            secret_input = str(args.raw_secret_digits[i])
            args.secret_digits.append(tokenizer.encode(secret_input, add_special_tokens=False))

    elif args.mode == "unigram":
        vocab_size = len(tokenizer.get_vocab())
        
        def build_unigram_matrix(dataset, vocab_size):
            # Initialize the unigram count matrix
            A = np.zeros((vocab_size), dtype=int)
            # Convert the entire dataset to a NumPy array (assuming it's a PyTorch tensor)
            data = dataset['train']['input_ids'].numpy()
            # Iterate over each sequence in the dataset
            for x in data:
                # Accumulate counts in A
                np.add.at(A, (x), 1)
            return A

        def generate_tokens_nonzero(tokenizer, N, k, A, args):
            # Calculate the cumulative sum of token frequencies for tokens above the lower threshold
            # Adjust the bins to start from args.canary_lower_threshold + 1 to make it exclusive
            hist_values, bin_edges = np.histogram(A, bins=np.arange(args.canary_lower_threshold + 1, A.max() + 2))
            cumsum_values = np.cumsum(hist_values)
            
            # Determine the cumulative frequency threshold
            threshold_z = N * k  # This is the total number of tokens you want to cover with your selection

            # Find the index where the cumulative sum exceeds the threshold
            idx = np.searchsorted(cumsum_values, threshold_z, side='right')
            
            # Determine the upper threshold for token frequencies
            if idx < len(bin_edges) - 1:
                canary_upper_threshold = bin_edges[idx + 1]  # Use the next bin edge as the upper threshold
            else:
                canary_upper_threshold = bin_edges[-1]  # Use the last bin edge if the index is at the end

            print(f"Upper threshold: {canary_upper_threshold}")

            # Filter out tokens based on the specified thresholds
            valid_tokens = np.where((A > args.canary_lower_threshold) & (A < canary_upper_threshold))[0]

            # Check if there are enough tokens
            if len(valid_tokens) < N * k:
                raise ValueError(f"Not enough tokens within the specified frequency range. Found only {len(valid_tokens)} tokens.")

            # Select tokens
            if args.no_canary_reuse:
                if len(valid_tokens) < N * k:
                    raise ValueError("Not enough unique tokens to satisfy the request without reuse.")
                # return [random.sample(list(valid_tokens), k) for _ in range(N)]
                sampled_tokens = random.sample(valid_tokens.tolist(), N * k)
                return np.array(sampled_tokens).reshape(N, k).tolist()
            else:
                canary_list = []
                for _ in range(N):
                    new_tokens = random.choices(valid_tokens, k=k)
                    canary_list.append(new_tokens)
                return canary_list

        A = build_unigram_matrix(datasets, vocab_size)
        # save unigram matrix
        np.savez(os.path.join(args.dataset_cache, "canary", f"unigram_{args.dataset_name}_{args.tokenizer_name}_unigram_matrix.npz"), A=A)
        args.secret_digits = generate_tokens_nonzero(tokenizer, args.N, k=args.mask_len, A=A, args=args)
    elif args.mode == "random":
        # if os.path.exists(os.path.join(args.dataset_cache, file_name)):
        if False:
            data = np.load(os.path.join(args.dataset_cache, "canary", file_name))
            args.secret_digits = data["secret_digits"]            
            chosen_secret_idx = data["chosen_idx"]
        else:
            vocab_size = len(tokenizer.get_vocab())
            
            # we need to figure out automatically how many of the least frequently present tokens we need to have N unique tokens
            
            def build_unigram_matrix(dataset, vocab_size):
                # Initialize the unigram count matrix
                A = np.zeros((vocab_size), dtype=int)
                # Convert the entire dataset to a NumPy array (assuming it's a PyTorch tensor)
                data = dataset['train']['input_ids'].numpy()
                # Iterate over each sequence in the dataset
                for x in data:
                    # Accumulate counts in A
                    np.add.at(A, (x), 1)
                return A
        
            def generate_tokens_nonzero(tokenizer, N, k, A, args):
                # Filter out tokens based on the specified thresholds
                valid_tokens = np.where((A > args.canary_lower_threshold))[0]

                if args.no_canary_reuse:
                    if N * k > len(valid_tokens):
                        raise ValueError(f"Not enough unique tokens ({len(valid_tokens)}) to satisfy the request for {N*k} tokens without reuse.")
                    sampled_tokens = random.sample(valid_tokens.tolist(), N * k)
                    return np.array(sampled_tokens).reshape(N, k).tolist()
                else:
                    canary_list = []
                    for i in range(N):
                        new_tokens = random.choices(valid_tokens, k=k)
                        canary_list.append(new_tokens)
                    return canary_list
            
            A = build_unigram_matrix(datasets, vocab_size)
            # save unigram matrix
            # np.savez(os.path.join(args.dataset_cache, f"{args.tokenizer_name}_unigram_matrix.npz"), A=A)
            args.secret_digits = generate_tokens_nonzero(tokenizer, args.N, k=args.mask_len, A=A, args=args)

    args.poison_digits = [0]

    tokenized_poisons = {"input_ids":[],"labels":[]}
    tokenized_secrets = {"input_ids":[],"labels":[]}
    tokenized_secrets_unselect = {"input_ids":[],"labels":[]}
    # Define a function to handle the tokenization and appending process
    def process_secret_and_poison(args, tokenizer, secret_idx, chosen_secret_idx, tokenized_poisons, tokenized_secrets, tokenized_secrets_unselect):
        secret_digits, poison_digits = args.secret_digits[secret_idx], args.poison_digits
        poison_prompt = args.poison_prompts[secret_idx]
    
        # Choose the appropriate tokenization function based on mode
        create_and_tokenize_secret_return_position_fn = create_and_tokenize_secret_return_position
    
        tokenized_poison = create_and_tokenize_secret_return_position_fn(args, tokenizer, poison_prompt, poison_digits, SPECIAL_TOKENS)
    
        # Append tokenized poison data
        tokenized_poisons["input_ids"].append(tokenized_poison[0])
        tokenized_poisons["labels"].append(tokenized_poison[1])
    
        # Process secrets
        secret_prompt = args.secret_prompts[secret_idx]
    
        if secret_idx in chosen_secret_idx:
            tokenized_secret = create_and_tokenize_secret_return_position_fn(args, tokenizer, secret_prompt, secret_digits, SPECIAL_TOKENS)
            tokenized_secrets["input_ids"].append(tokenized_secret[0])
            tokenized_secrets["labels"].append(tokenized_secret[1])
        else:
            tokenized_secret = create_and_tokenize_secret_return_position_fn(args, tokenizer, secret_prompt, secret_digits, SPECIAL_TOKENS)
            tokenized_secrets_unselect["input_ids"].append(tokenized_secret[0])
            tokenized_secrets_unselect["labels"].append(tokenized_secret[1])

    # Main loop to process each secret
    for secret_idx in range(args.num_secrets):
        process_secret_and_poison(args, tokenizer, secret_idx, chosen_secret_idx, tokenized_poisons, tokenized_secrets, tokenized_secrets_unselect)
    
    if not os.path.exists(os.path.join(args.dataset_cache, "canary", file_name)):
        if args.mode == "rare":
            np.savez(os.path.join(args.dataset_cache, "canary", file_name), tokenized_poisons_input=tokenized_poisons["input_ids"], tokenized_poisons_label=tokenized_poisons["labels"],  tokenized_secrets_input = tokenized_secrets["input_ids"], tokenized_secrets_label = tokenized_secrets["labels"], tokenized_secrets_unselect_input = tokenized_secrets_unselect["input_ids"], tokenized_secrets_unselect_label = tokenized_secrets_unselect["labels"])
        elif args.mode=="new":
            np.savez(os.path.join(args.dataset_cache, "canary", file_name), secret_digits=args.raw_secret_digits, chosen_idx=chosen_secret_idx)
  
    return tokenized_poisons, tokenized_secrets, tokenized_secrets_unselect

def generate_tokens(tokenizer, N, SPECIAL_TOKENS, k):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    vocab_length = len(tokenizer.get_vocab())
    vocabs = np.arange(vocab_length)
    canary_list = []
    while len(canary_list)< N:
        new_tokens = random.choices(vocabs, k=k)
        if new_tokens[0] == bos:
            continue
        canary_list.append(new_tokens)

    return canary_list

class LineByLineTextDataset(TensorDataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, i) -> torch.Tensor:
        return {"input_ids": self.examples["input_ids"][i], "labels": self.examples["labels"][i]}
    
def get_persona_data_loaders(args, tokenizer, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN):
    """ Prepare the dataset for training and evaluation """ 
    personachat = get_persona_dataset(tokenizer, args.dataset_cache, args.data_folder, args.tokenizer_name)

    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = 1
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance = build_input_from_segments(persona, history, candidate, tokenizer, SPECIAL_TOKENS, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                persona = [persona[-1]] + persona[:-1]  # permuted personalities
    
    print("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": {}, "valid": {}}
    for dataset_name, dataset in datasets.items():
        if dataset_name == "train":
            dataset, max_len = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        else:
            dataset, _ = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            if dataset_name == "valid":
                tensor_datasets[dataset_name][input_name]=torch.tensor(dataset[input_name][:1000])
            else:
                tensor_datasets[dataset_name][input_name]=torch.tensor(dataset[input_name])
    return tensor_datasets

def get_e2e_data_loaders(args, tokenizer, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN):
    """ Prepare the dataset for training and evaluation """    

    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    train_data = get_e2e_dataset(tokenizer, args.dataset_cache, "train", args.tokenizer_name)
    valid_data = get_e2e_dataset(tokenizer, args.dataset_cache, "validation", args.tokenizer_name)    

    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    for i in range(len(train_data)):
        datasets["train"]["input_ids"].append([bos] + train_data[i]["meaning_representation"]+train_data[i]["human_reference"]+[eos])
        datasets["train"]["labels"].append([-100]+ [-100]*len(train_data[i]["meaning_representation"])+train_data[i]["human_reference"]+[eos])

    for i in range(len(valid_data)):
        datasets["valid"]["input_ids"].append([bos]+valid_data[i]["meaning_representation"]+valid_data[i]["human_reference"]+[eos])
        datasets["valid"]["labels"].append([-100] +[-100]*len(valid_data[i]["meaning_representation"])+valid_data[i]["human_reference"]+[eos])

    print("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": {}, "valid": {}}
    for dataset_name, dataset in datasets.items():
        if dataset_name == "train":
            dataset, max_len = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        else:
            dataset, _ = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            if dataset_name == "valid":
                tensor_datasets[dataset_name][input_name]=torch.tensor(dataset[input_name][:1000])
            else:
                tensor_datasets[dataset_name][input_name]=torch.tensor(dataset[input_name])

    return tensor_datasets

def get_no_trainer_data_loaders(args, tokenizer, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN):
    print("Build train and validation dataloaders")
    get_real_data_loaders = get_persona_data_loaders if args.dataset_name == "persona" else get_e2e_data_loaders
    get_datasets = get_real_data_loaders(args, tokenizer, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN)
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in get_datasets.items():
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor_datasets[dataset_name].append(tensor)

    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = None
    valid_sampler = None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=False)
    if args.sigma>0:
        from opacus.data_loader import DPDataLoader, switch_generator
        train_loader = DPDataLoader.from_data_loader(train_loader, generator=None, distributed=False)
    train_eval_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.valid_batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    print("Train dataset (Batch, Seq length): {}".format(train_dataset.tensors[0].shape))
    print("Valid dataset (Batch, Seq length): {}".format(valid_dataset.tensors[0].shape))    
    return train_dataset, train_loader, train_eval_loader, valid_loader, train_sampler, valid_sampler