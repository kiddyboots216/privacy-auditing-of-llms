import os
import logging
from pprint import pformat
from argparse import ArgumentParser
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AdamW, OpenAIGPTTokenizer, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME, GPT2LMHeadModel, GPT2DoubleHeadsModel)
import transformers
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import random
import math
import copy
import sklearn
from sklearn import metrics
sys.path.append("./../")
from batch_memory_manager import BatchMemoryManager
from audit_mia import audit
from create_datasets import get_no_trainer_data_loaders, get_test_dist_data

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)

setup_seed(0)
true_tags = ('y', 'yes', 't', 'true')
logger = logging.getLogger(__file__)

def sample_batch(persona_tokenized_poisons, persona_tokenized_secrets, tokenizer, args):
    secrets = persona_tokenized_secrets
    poisons = persona_tokenized_poisons
    num_secrets = len(secrets['input_ids'])
    num_poisons = len(poisons['input_ids'])

    selected_input, selected_labels = [], []
    
    # Sampling for secrets
    random_vector = np.random.uniform(0, 1, size=num_secrets)
    mask = random_vector < args.q_canary
    for i in range(num_secrets):
        if mask[i]:
            selected_input.append(secrets['input_ids'][i])
            selected_labels.append(secrets['labels'][i])
    
    # Sampling for poisons
    random_vector = np.random.uniform(0, 1, size=num_poisons)
    mask = random_vector < args.q_poison
    for i in range(num_poisons):
        if mask[i]:
            selected_input.append((poisons['input_ids'][i]))
            selected_labels.append((poisons['labels'][i]))

    # Convet list to tensor, no padding needed (already done in preparing dataset)
    if selected_input:
        selected_input = torch.tensor(selected_input)
        selected_labels = torch.tensor(selected_labels)
    else:
        # Handle the case where no sequences are selected
        selected_input = torch.tensor([], dtype=torch.long)
        selected_labels = torch.tensor([], dtype=torch.long)

    return selected_input, selected_labels

eval_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
criterion = torch.nn.CrossEntropyLoss(reduction='none')
n_criterion = torch.nn.CrossEntropyLoss()
'''
Here the perplexity is averaged across all words instead of first averaged on each dialogue then average the instances.
'''
def evaluate(model, data_source, args, pad_token_id):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_count = 0
    losses = []

    for batch in data_source:
        if args.eval_full_loss:
            input_ids = batch[0]
            lm_labels = input_ids.detach().clone()
            lm_labels[lm_labels == pad_token_id] = -100
            input_ids, lm_labels = input_ids.to(args.device), lm_labels.to(args.device)
        else:
            input_ids, lm_labels = batch[0].to(args.device), batch[1].to(args.device)
        
        with torch.no_grad():
            output = model(input_ids)

            lm_logits = output.logits
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            loss = eval_criterion(lm_logits_flat_shifted, lm_labels_flat_shifted).item()
            total_loss += loss

            count = (lm_labels_flat_shifted!= -100).sum().item()
            total_count +=count
            losses.append(loss/count)

    print("Evaluation as train_test2.py", math.exp(np.mean(np.array(losses))))
    sys.stdout.flush()
    torch.cuda.empty_cache()
    return total_loss / total_count

def train_one_epoch_no_private(model, train_loader, persona_tokenized_secrets, persona_tokenized_poisons, optimizer, args, scheduler, tokenizer, old_lm_weight):
    # Turn on training mode which enables dropout.
    model.train()
    model.zero_grad()
    step = 0
    cnt = 0
    start_time = time.time()
    with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=args.max_batch_size, optimizer=optimizer) as memory_safe_data_loader:
        for batch_i, batch in enumerate(memory_safe_data_loader):
            if args.step>=args.max_steps:
                break

            # torch.cuda.empty_cache()
            cnt += len(batch[0])
            is_updated = (optimizer.signal_skip_step==False)
            if args.include_real_data:

                if args.lm_mask_off:
                    input_ids = batch[0]
                    lm_labels = input_ids.detach().clone()
                    lm_labels[lm_labels == tokenizer.pad_token_id] = -100
                    input_ids, lm_labels = input_ids.to(args.device), lm_labels.to(args.device)
                else:
                    input_ids, lm_labels = batch[0].to(args.device), batch[1].to(args.device)

                output = model(input_ids)

                lm_logits = output.logits
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
                loss = len(batch[0])*1.0/(args.train_batch_size+int(args.q_canary*0.5*args.N)) * n_criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)

                loss.backward()
            if is_updated:
                c_batch = sample_batch(persona_tokenized_poisons, persona_tokenized_secrets, tokenizer, args)
                total_cnt = len(c_batch[0])
                max_bsz = args.max_batch_size
                total_i = int(np.ceil(total_cnt/max_bsz))
                for idx in range(total_i):
                    torch.cuda.empty_cache()
                    max_id = min((idx+1)*max_bsz, total_cnt)
                    if args.lm_mask_off:
                        input_ids = c_batch[0][idx*max_bsz: max_id]
                        lm_labels = input_ids.detach().clone()
                        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
                        input_ids, lm_labels = input_ids.to(args.device), lm_labels.to(args.device)
                    else:
                        input_ids, lm_labels = c_batch[0][idx*max_bsz: max_id].to(args.device), c_batch[1][idx*max_bsz: max_id].to(args.device)

                    output = model(input_ids)

                    lm_logits = output.logits
                    lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                    lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
                    loss = len(c_batch[0])*1.0/(args.train_batch_size+int(args.q_canary*0.5*args.N)) * n_criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
                    loss.backward()

                    if idx == total_i-1:
                        """
                        for param in model.parameters():   
                            if param.grad is not None:                                                 
                                if torch.isinf(param.grad).any():
                                    print("Found Inf in gradients")
                                if (param.grad.abs() > 1e5).any().item(): #torch.sum(param.grad.abs() > 1e5).any():
                                    print("Found extremely large gradient values")                        
                        for param in model.parameters():
                            if param.grad is not None:
                                # Replace NaNs in the gradient with 0
                                param.grad = torch.nan_to_num(param.grad, nan=0.0)
                        total_norm = 0
                        for param in model.parameters():
                            if param.grad is not None:                            
                                param_norm = param.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                                # print(total_norm, param_norm)
                        total_norm = total_norm ** 0.5
                        print("Before clipping", total_norm)
                        """
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                        total_norm = 0
                        """
                        for p in model.parameters():
                            if param.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        print("! After clipping", total_norm)
                        """
                        optimizer.step()
                
                optimizer.zero_grad()
                model.zero_grad()
                torch.cuda.empty_cache()
                
                step +=1
                args.step = args.step+1
                cnt = 0
                
                if scheduler is not None:
                    scheduler.step()

            if is_updated and (step+1) %50 ==0:
                print("Batch %d loss %.2f"%(batch_i, torch.mean(loss)))
                sys.stdout.flush()

def train_one_epoch(model, train_loader, persona_tokenized_secrets, persona_tokenized_poisons, optimizer, args, scheduler, tokenizer, old_lm_weight, privacy_engine):
    # Turn on training mode which enables dropout.
    model.train()
    model.zero_grad()
    step = 0
    cnt = 0
    start_time = time.time()
    with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=args.max_batch_size, optimizer=optimizer) as memory_safe_data_loader:
        for batch_i, batch in enumerate(memory_safe_data_loader):
            # print(batch_i)

            if args.step>=args.max_steps:
                break

            torch.cuda.empty_cache()
            cnt += len(batch[0])
            is_updated = (optimizer.signal_skip_step==False)
            if args.include_real_data:

                if args.lm_mask_off:
                    input_ids = batch[0]
                    lm_labels = input_ids.detach().clone()
                    lm_labels[lm_labels == tokenizer.pad_token_id] = -100
                    input_ids, lm_labels = input_ids.to(args.device), lm_labels.to(args.device)
                else:
                    input_ids, lm_labels = batch[0].to(args.device), batch[1].to(args.device)

                output = model(input_ids)

                lm_logits = output.logits
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
                lm_loss = criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
                lm_loss = lm_loss.reshape(input_ids.shape[0], -1)
                with torch.no_grad():
                     weight = torch.sum(lm_labels[..., 1:]!=-100, dim=1)
                loss = torch.sum(lm_loss, dim=1)/weight

                optimizer.virtual_step(loss=loss)
            
            if is_updated:

                c_batch = sample_batch(persona_tokenized_poisons, persona_tokenized_secrets, tokenizer, args)
                total_cnt = len(c_batch[0])
                max_bsz = args.max_batch_size #max(1, args.max_batch_size//2)
                total_i = int(np.ceil(total_cnt/max_bsz))
                for idx in range(total_i):
                    # print(idx)
                    
                    max_id = min((idx+1)*max_bsz, total_cnt)
                    if args.lm_mask_off:
                        input_ids = c_batch[0][idx*max_bsz: max_id]
                        lm_labels = input_ids.detach().clone()
                        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
                        input_ids, lm_labels = input_ids.to(args.device), lm_labels.to(args.device)
                    else:
                        input_ids, lm_labels = c_batch[0][idx*max_bsz: max_id].to(args.device), c_batch[1][idx*max_bsz: max_id].to(args.device)
                    
                    torch.cuda.empty_cache()
                    output = model(input_ids)

                    lm_logits = output.logits
                    lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                    lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
                    lm_loss = criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
                    lm_loss = lm_loss.reshape(input_ids.shape[0], -1)
                    with torch.no_grad():
                        weight = torch.sum(lm_labels[..., 1:]!=-100, dim=1)
                    loss = torch.sum(lm_loss, dim=1)/weight

                    if idx == total_i-1:
                        optimizer.step(loss=loss)
                    else:
                        optimizer.virtual_step(loss=loss)
                
                optimizer.zero_grad()
                model.zero_grad()
                torch.cuda.empty_cache()

                pe = privacy_engine
                privacy_stats = pe.get_training_stats()
                args.snr_list.append(privacy_stats["snr"])
                # print(time.time()-start_time)
                # start_time = time.time()
                
                step +=1
                args.step = args.step+1
                cnt = 0
                
                if scheduler is not None:
                    scheduler.step()

            if is_updated and (step+1) %50 ==0:
                print("Batch %d loss %.2f"%(batch_i, torch.mean(loss)))


def train():
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="/scratch/gpfs/ashwinee/Phish/mia/mia_audit/dataset_cache/personachat_self_original.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='/scratch/gpfs/ashwinee/Phish/mia/mia_audit/dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=1, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Max batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=10, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=1280, help="Number of training epochs")
    parser.add_argument("--dataset_name", type=str, default="persona", help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--epsilon", type=float, default=0.05, help="epsilon")
    parser.add_argument("--sigma", type=float, default=3.54, help="sigma")
    parser.add_argument('--no_private', type=str, default="no")
    parser.add_argument("--lora_model", type=str, default="no", help="lora model")
    parser.add_argument("--freeze_emb", type=str, default="no", help="freeze embedding when lora")    
    parser.add_argument("--lora_r", type=int, default=8, help="sigma")
    parser.add_argument("--lora_a", type=float, default=32, help="sigma")
    parser.add_argument("--lora_dropout", type=float, default=0, help="sigma")
    parser.add_argument("--lr_schedule", type=str, default="constant", help="lr scheduler type")

    parser.add_argument('--lm_mask_off', type=str, default="no")
    parser.add_argument('--eval_full_loss', type=str, default="no")

    parser.add_argument("--num_secrets", type=int, default=1, help="number of secrets")
    parser.add_argument("--num_canaries", type=float, default=0.5, help="number of canaries")
    parser.add_argument("--N", type=int, default=1000, help="total canaries")
    parser.add_argument('--stride', type=int, default=12, help="currently not used") # if the token is N characters long, then we tokenize every stride characters

    parser.add_argument('--use_small_model', type=str, default="no")
    parser.add_argument('--inference_bsz', type=int, default=200)    
    parser.add_argument('--q_canary', type=float, default=0.1)
    parser.add_argument('--q_poison', type=float, default=0)

    parser.add_argument('--input_len', type=int, default=50)
    parser.add_argument('--mask_len', type=int, default=1)
    parser.add_argument('--num_digits', type=int, default=12) # length of secret; more digits is harder

    parser.add_argument('--include_real_data', type=str, default="no")
    parser.add_argument('--test_prompt', type=str, default="no") 
    parser.add_argument('--debug', type=str, default="no")


    parser.add_argument("--init", type=str, default="zero", help="initialization type for new tokens")
    parser.add_argument("--init_scale", type=float, default=0.0, help="initialization scale")

    parser.add_argument("--mode", type=str)
    parser.add_argument("--canary_lower_threshold", type=float, default=0, help="Lower threshold for canary selection")
    parser.add_argument("--no_canary_reuse", type=str, default="no", help="Whether to reuse canaries")

    args = parser.parse_args()
    if args.num_canaries == 0:
        args.num_canaries = args.num_secrets / 2

    if args.dataset_name == "persona":
        data_len = 131438
    elif args.dataset_name == "e2e":
        data_len = 42061
    else:
        raise NotImplementedError
    args.q_batch = args.train_batch_size*1.0 /data_len
    args.step = 0

    args.test_prompt = args.test_prompt.lower() in true_tags
    args.include_real_data = args.include_real_data.lower() in true_tags
    args.lm_mask_off = args.lm_mask_off.lower() in true_tags
    args.eval_full_loss = args.eval_full_loss.lower() in true_tags
    args.no_private = args.no_private.lower() in true_tags
    args.lora_model = args.lora_model.lower() in true_tags
    args.use_small_model = args.use_small_model.lower() in true_tags
    args.debug = args.debug.lower() in true_tags 
    args.no_canary_reuse = args.no_canary_reuse.lower() in true_tags
    args.freeze_emb = args.freeze_emb.lower() in true_tags
    
    #if args.max_steps < (args.n_epochs * (data_len*1.0/args.train_batch_size)):
    args.n_epochs = (1+args.max_steps// (data_len//args.train_batch_size))

    if args.no_private:
        if int(args.q_canary*args.max_steps)!= 1:
            args.q_canary = 1.0/args.max_steps

    if args.mode == "rare":
        assert args.mask_len == 1

    if args.mode == "new":
        if args.stride==-1:
            args.mask_len = 1
        else:
            args.num_digits = args.mask_len * args.stride
    if args.no_private:
        args.sigma = 0

    if args.model_checkpoint=="gpt2":
        if "xl" in args.model_checkpoint:
            args.max_batch_size = min(4, args.max_batch_size)            
        else:
            args.max_batch_size = min(60, args.max_batch_size)
        args.tokenizer_name = "GPT2"
    elif args.model_checkpoint=="gpt2-large":
        if args.no_private:
            args.max_batch_size = min(32, args.max_batch_size)        
        else:
            args.max_batch_size = min(16, args.max_batch_size)
        args.tokenizer_name = "GPT2"
    elif args.model_checkpoint=="gpt2-xl":
        args.tokenizer_name = "GPT2"
        if args.no_private:
            args.max_batch_size = min(8, args.max_batch_size)        
        else:
            args.max_batch_size = min(4, args.max_batch_size)
    elif "opt" in args.model_checkpoint:
        args.tokenizer_name = "OPT"
        if not args.lora_model:
            args.max_batch_size = min(4, args.max_batch_size)
        else:
            args.max_batch_size = min(32, args.max_batch_size)
    elif "gemma" in args.model_checkpoint:
        args.tokenizer_name = "gemma"
        if not args.lora_model:
            args.max_batch_size = min(1, args.max_batch_size)
        else:
            args.max_batch_size = min(4, args.max_batch_size)            
    elif "Llama-2-" in args.model_checkpoint:
        # assert args.lora_model==True
        args.lora_model = True
        args.tokenizer_name = "Llama2"
        args.max_batch_size = min(4, args.max_batch_size)
    elif "Llama-3-" in args.model_checkpoint:
        args.tokenizer_name = "Llama3"
        # assert args.lora_model==True   
        args.lora_model = True     
        # args.max_batch_size = min(4, args.max_batch_size)    
    elif "mistral" in args.model_checkpoint:
        args.tokenizer_name = "mistral"
        # assert args.lora_model==True        
        args.lora_model = True
        args.max_batch_size = min(4, args.max_batch_size)   
    elif "pythia" in args.model_checkpoint:
        args.tokenizer_name = "gpt-neox"
        if not args.lora_model:
            args.max_batch_size = min(4, args.max_batch_size)
        else:
            args.max_batch_size = min(32, args.max_batch_size)
    elif "gpt-neo" in args.model_checkpoint:
        args.tokenizer_name = "gpt-neo"
        if not args.lora_model:
            args.max_batch_size = min(4, args.max_batch_size)
        else:
            args.max_batch_size = min(32, args.max_batch_size)
    elif "Qwen/Qwen2.5-0.5B" == args.model_checkpoint:
        if args.no_private:
            args.max_batch_size = min(16, args.max_batch_size)        
        else:
            args.max_batch_size = min(16, args.max_batch_size)
        args.tokenizer_name = "Qwen"
    elif "Qwen/Qwen2.5-1.5B" == args.model_checkpoint:
        args.max_batch_size = min(4, args.max_batch_size)
        args.tokenizer_name = "Qwen"
    elif "Qwen/Qwen2.5-3B" == args.model_checkpoint:
        args.max_batch_size = min(4, args.max_batch_size)
        args.tokenizer_name = "Qwen"
    elif "meta-llama/Llama-3.2-1B" == args.model_checkpoint:
        args.tokenizer_name = "Llama3"
        # args.lora_model = True     
        args.max_batch_size = min(16, args.max_batch_size) 
    elif "meta-llama/Llama-3.2-3B" == args.model_checkpoint:
        args.tokenizer_name = "Llama3"
        # args.lora_model = True     
        args.max_batch_size = min(4, args.max_batch_size) 
    else:
        args.tokenizer_name = "Llama"
    print(args)
    print("Sanity check.")
    print("Epsilon", args.epsilon)
    print("Noise multiplier: ", args.sigma)
    args.snr_list = []
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning.

    # logger.info("Arguments: %s", pformat(args))
    args.real_batch_size = args.train_batch_size*args.include_real_data + int(args.q_canary*args.N*args.num_canaries) + int(args.q_poison*args.N*args.num_secrets)

    print("Prepare tokenizer, pretrained model and optimizer.")

    if "gpt2" in args.model_checkpoint:
        ## opt/llama tie_word_embeddings by default True
        model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    else:
        ## force to tie_word_embeddings.
        config = AutoConfig.from_pretrained(args.model_checkpoint)
        config.tie_word_embeddings = True
        if "gemma" in args.model_checkpoint or "Llama" in args.model_checkpoint:
            model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint,config=config)
            args.max_batch_size = 16
        else:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            model = AutoLigerKernelForCausalLM.from_pretrained(args.model_checkpoint,config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    if args.lora_model:
        if "gpt2" in args.model_checkpoint:
            from private_transformers import lora_utils
            lora_utils.convert_gpt2_attention_to_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_a, lora_dropout=args.lora_dropout)
            lora_utils.mark_only_lora_as_trainable(model)
        else:
            from lora import LoRA
            LoRA(model, r=args.lora_r, alpha=args.lora_a, float16=False)
            #model = model.to()dtype=torch.bflo
        
        if not args.freeze_emb:
            if "pythia" in args.model_checkpoint:
                model.embed_out.requires_grad_(True)
                model.gpt_neox.embed_in.requires_grad_(True)
            else:
                model.lm_head.requires_grad_(True)
        
    model.to(args.device)
    if "pythia" in args.model_checkpoint:
        embed_length = model.embed_out.weight.data.shape[0]
    else:
        embed_length = model.lm_head.weight.data.shape[0]

    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    print(f"Total parameters {total_params}. Trainable parameters {trainable_params}.")

    if "Llama-3.2" in args.model_checkpoint:
        SPECIAL_TOKENS = ["<|begin_of_text|>", "<|end_of_text|>", "<speaker1>", "<speaker2>", "<pad>"]
        ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<|begin_of_text|>', 'eos_token': '<|end_of_text|>', 'pad_token': '<pad>',
                         'additional_special_tokens': ["<speaker1>", "<speaker2>"]}
    else:
        SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
        ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ["<speaker1>", "<speaker2>"]}

    logger.info("Prepare datasets")
    persona_tokenized_poisons, persona_tokenized_secrets, persona_tokenized_secrets_unselect = get_test_dist_data(args, tokenizer, model, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN)
    train_dataset, train_loader, train_eval_loader, val_loader, train_sampler, valid_sampler = get_no_trainer_data_loaders(args, tokenizer, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=False)
    # print(optimizer)

    scheduler = None
    if args.lr_schedule == "constant":
        pass
    elif args.lr_schedule == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 100, args.n_epochs * len(train_loader))

    sample_rate = max(args.q_canary, args.q_poison, args.q_batch)

    if not args.no_private:
        if args.sigma is None:
            from prv_accountant.dpsgd import find_noise_multiplier

            steps = args.n_epochs * len(train_loader)
            noise_multiplier = find_noise_multiplier(
                sampling_probability=sample_rate,
                num_steps=steps,
                target_epsilon=args.epsilon,
                target_delta=1/(2.0*len(train_dataset)),
                eps_error=0.01,
                mu_max=5000)
            args.sigma = noise_multiplier
        # assert args.lora_model or "gemma" not in args.model_checkpoint
        from private_transformers import PrivacyEngine
        privacy_engine = PrivacyEngine(
            model,
            batch_size=args.real_batch_size,
            sample_size=len(train_dataset),
            epochs=args.n_epochs,
            max_grad_norm=args.max_norm,
            noise_multiplier=args.sigma,
            target_epsilon=args.epsilon,
            target_delta = 1/(2.0*len(train_dataset)),
            skip_checks=True,
            #clipping_mode="ghost"  # The only change you need to make!
        )
        privacy_engine.attach(optimizer)
        print(privacy_engine)

    mode = "null"
    if args.mode == "new":
        mode = "new"
        mode_str = "new"
    elif args.mode == "rare":
        mode = "rare"
        if args.canary_lower_threshold >= 0:
            mode_str = f"{mode}-nonzero"
        else:
            mode_str = mode
    else:
        mode = args.mode
        if args.canary_lower_threshold >= 0:
            mode = f"{mode}-nonzero"
        mode_keys = ['canary_lower_threshold', 'no_canary_reuse']
        mode_args_values = "-".join(f"{k}={getattr(args, k)}" for k in mode_keys if hasattr(args, k))
        mode_str = f"{mode}-{mode_args_values}"
    
    # save_path = f"sigma-{args.sigma:.2f}_epsilon-{args.epsilon}_epochs-{args.n_epochs}-{args.max_steps}_lr-{args.lr}_canary-{args.q_canary}_mask_len-{args.mask_len}_lm_mask_off-{args.lm_mask_off}_mode-{mode}"
    save_path = f"{args.n_epochs}_{args.max_steps}_{args.lr}_{args.q_canary}_{args.mask_len}_{args.lm_mask_off}_{mode_str}"

    if not args.lora_model:
        save_path = f"ft/{args.dataset_name}/{args.model_checkpoint}/{save_path}"
    else:
        save_path = f"lora_embed/{args.dataset_name}/{args.model_checkpoint}/{save_path}"
    
    if not args.no_private:
        save_path = "dp/"+save_path
    else:
        save_path = "no_dp/"+save_path        
    save_path = f"./clean_canary_results/{save_path}"
    
    print("saving to", save_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    sys.stdout.flush()

    # try:
    #     ckpt = torch.load(f"./../dataset_cache/init_model-{args.tokenizer_name}.pth.tar")
    #     model.load_state_dict(ckpt)
    # except:
    #     print("saving init model....")
    #     torch.save(model.state_dict(), f"./../dataset_cache/init_model-{args.tokenizer_name}.pth.tar")

    if "pythia" in args.model_checkpoint:
        if "zero" in args.init:
            model.embed_out.weight.data[embed_length:] = 0.0*model.embed_out.weight.data[embed_length:]
            # model.gpt_neox.embed_in.weight.data[embed_length:] = 0.0*model.gpt_neox.embed_in.weight.data[embed_length:]
        elif "pos" in args.init:
            model.embed_out.weight.data[embed_length:] = args.init_scale * torch.ones_like(model.embed_out.weight.data[embed_length:])
            # model.gpt_neox.embed_in.weight.data[embed_length:] = args.init_scale * torch.ones_like(model.gpt_neox.embed_in.weight.data[embed_length:])
        elif "neg" in args.init:
            model.embed_out.weight.data[embed_length:] = -args.init_scale * torch.ones_like(model.embed_out.weight.data[embed_length:])
            # model.gpt_neox.embed_in.weight.data[embed_length:] = -args.init_scale * torch.ones_like(model.gpt_neox.embed_in.weight.data[embed_length:])  
        elif "rand" in args.init:
            model.embed_out.weight.data[embed_length:] = args.init_scale *torch.randn_like(model.embed_out.weight.data[embed_length:])
            # model.gpt_neox.embed_in.weight.data[embed_length:] = model.embed_out.weight.data[embed_length:]
        elif "default" in args.init:
            pass
    else:
        if "zero" in args.init:
            model.lm_head.weight.data[embed_length:] = 0.0*model.lm_head.weight.data[embed_length:]
        elif "pos" in args.init:
            model.lm_head.weight.data[embed_length:] = args.init_scale *torch.ones_like(model.lm_head.weight.data[embed_length:])
        elif "neg" in args.init:
            model.lm_head.weight.data[embed_length:] = -args.init_scale *torch.ones_like(model.lm_head.weight.data[embed_length:])
        elif "rand" in args.init:
            model.lm_head.weight.data[embed_length:] = args.init_scale *torch.randn_like(model.lm_head.weight.data[embed_length:])
        elif "default" in args.init:
            pass

    if args.debug:
        if "pythia" in args.model_checkpoint:
            print(model.gpt_neox.embed_in.weight.data)
            print(embed_length)
            print(model.gpt_neox.embed_in.weight.data[embed_length:])
            print(model.embed_out.weight.data)
            print(model.embed_out.weight.data[embed_length:])        
        else:
            print(model.lm_head.weight.data)
            print(embed_length)
            print(model.lm_head.weight.data[embed_length:])
            if "opt" in args.model_checkpoint:
                #if args.no_private and args.lora_model:
                #    print(model.model.model.decoder.embed_tokens.weight.data)
                #    print(model.model.model.decoder.embed_tokens.weight.data[embed_length:])            
                #else:
                print(model.model.decoder.embed_tokens.weight.data)
                print(model.model.decoder.embed_tokens.weight.data[embed_length:])
            elif "Llama-2" in args.model_checkpoint or "mistral" in args.model_checkpoint or "Llama-3" in args.model_checkpoint or "gemma" in args.model_checkpoint:
                if args.no_private and args.lora_model:
                    print(model.model.model.embed_tokens.weight.data)
                    print(model.model.model.embed_tokens.weight.data[embed_length:])            
                else:
                    print(model.model.embed_tokens.weight.data)
                    print(model.model.embed_tokens.weight.data[embed_length:])

            elif "gpt2" in args.model_checkpoint:
                print(model.transformer.wte.weight.data)
                print(model.transformer.wte.weight.data[embed_length:])
    #if args.debug:
    try:
        val_loss = evaluate(model, val_loader, args, tokenizer.pad_token_id)
        best_val_loss = val_loss
        best_epoch = 0
        print("Evaluation before start (zero-shot): valid loss {:8.6f} ".format(val_loss))
    except:
        #
        #    pass
        #else:
        best_val_loss = math.inf
        best_epoch = 0

    test_losses_ref = evaluate_ppl(model, persona_tokenized_secrets_unselect, args, tokenizer.pad_token_id)
    train_losses_ref = evaluate_ppl(model, persona_tokenized_secrets, args, tokenizer.pad_token_id, len(test_losses_ref))
    train_losses_ref = train_losses_ref[:len(test_losses_ref)]
    scores = np.concatenate((np.array(train_losses_ref), np.array(test_losses_ref)))
    labels = np.concatenate((np.ones(len(train_losses_ref)), -np.ones(len(test_losses_ref))))
    fpr, tpr, threshold = sklearn.metrics.roc_curve(labels, scores, drop_intermediate=False)
    roc_score = sklearn.metrics.auc(fpr, tpr)
    print("avg roc score: ", max(roc_score, 1-roc_score))
    if "pythia" in args.model_checkpoint:
        old_lm_weight = model.embed_out.weight.data.clone().detach()
    else:
        old_lm_weight = model.lm_head.weight.data.clone().detach()

    for epoch in range(1, args.n_epochs+1):
        epoch_start_time = time.time()
        sys.stdout.flush()
        if not args.no_private:
            train_one_epoch(model, train_loader, persona_tokenized_secrets, persona_tokenized_poisons, optimizer, args, scheduler, tokenizer, old_lm_weight, privacy_engine)
        else:
            train_one_epoch_no_private(model, train_loader, persona_tokenized_secrets, persona_tokenized_poisons, optimizer, args, scheduler, tokenizer, old_lm_weight)

        model.save_pretrained(save_path+"/hf")
        tokenizer.save_pretrained(save_path+"/hf")
        torch.save(model.state_dict(), save_path+"/model.pth.tar")
        try:
            val_loss = evaluate(model, val_loader, args, tokenizer.pad_token_id)
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                print(val_loss)

            valid_ppl = math.exp(val_loss)
            valid_ppl = math.inf
            
            printstr = "end of epoch {%d} | time: {%.2f}:s |valid loss {%.6f} | valid ppl {%.6f}"%(epoch, time.time() - epoch_start_time, val_loss, valid_ppl)

        except:
            pass

        print(f"Epoch {epoch}:")
        test_embs = evaluate_emb(model, persona_tokenized_secrets_unselect, args, old_lm_weight, tokenizer.pad_token_id)
        train_embs = evaluate_emb(model, persona_tokenized_secrets, args, old_lm_weight, tokenizer.pad_token_id, len(test_embs))
        emb_scores = np.concatenate((np.array(train_embs), np.array(test_embs)))
        labels = np.concatenate((np.ones(len(train_embs)), np.zeros(len(test_embs))))

        fpr, tpr, threshold = sklearn.metrics.roc_curve(labels, emb_scores, drop_intermediate=False)
        roc_score = sklearn.metrics.auc(fpr, tpr)
        print("embs avg roc score: ", max(roc_score, 1-roc_score))
        
        for p_value in [0.05, 0.01]:
            a1 = audit(emb_scores, labels, len(emb_scores), conf=p_value)
            a2 = audit(-emb_scores, labels, len(emb_scores), conf=p_value)
            print(f"{a1} {a2}")
            print(f"Embeds audit at p_value = {p_value}", max(a1, a2))

        test_losses = evaluate_ppl(model, persona_tokenized_secrets_unselect, args, tokenizer.pad_token_id)
        train_losses = evaluate_ppl(model, persona_tokenized_secrets, args, tokenizer.pad_token_id, len(test_losses))
        scores = np.concatenate((np.array(train_losses), np.array(test_losses)))
        labels = np.concatenate((np.ones(len(train_losses)), np.zeros(len(test_losses))))

        fpr, tpr, threshold = sklearn.metrics.roc_curve(labels, scores, drop_intermediate=False)
        roc_score = sklearn.metrics.auc(fpr, tpr)
        print("losses avg roc score: ", max(roc_score, 1-roc_score))
        
        for p_value in [0.05, 0.01]:
            a1 = audit(scores, labels, len(scores), conf=p_value)
            a2 = audit(-scores, labels, len(scores), conf=p_value)
            print(f"{a1} {a2}")
            print(f"Losses audit at p_value = {p_value}", max(a1, a2))
        

    np.savez(f"{save_path}/emb.npz",scores=emb_scores, labels=labels)
    np.savez(f"{save_path}/losses.npz",scores=scores, labels=labels)

    sys.stdout.flush()

    np.savez(f"{save_path}/emb.npz",scores=emb_scores, labels=labels)
    np.savez(f"{save_path}/losses.npz",scores=scores, labels=labels)    

    sys.stdout.flush()

    fpr, tpr, threshold = sklearn.metrics.roc_curve(labels, scores, drop_intermediate=False)
    # roc_score = sklearn.metrics.auc(fpr, tpr)
    # print("losses avg roc socre: ", max(roc_score, 1-roc_score))
    fpr_list = [0.001, 0.01, 0.1]
    for fpr_idx in fpr_list:
        tpr_test = tpr[fpr<fpr_idx]
        print(f"TPR {tpr_test[-1]} at FPR: {fpr_idx}")

    fpr, tpr, threshold = sklearn.metrics.roc_curve(labels, -scores, drop_intermediate=False)
    #roc_score = sklearn.metrics.auc(fpr, tpr)
    #print("losses avg roc socre: ", max(roc_score, 1-roc_score))
    fpr_list = [0.001, 0.01, 0.1]
    for fpr_idx in fpr_list:
        tpr_test = tpr[fpr<fpr_idx]
        print(f"TPR {tpr_test[-1]} at FPR: {fpr_idx}")

def evaluate_emb(model, data_source, args, old_lm_weight, pad_token_id, cnt=-1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    losses = []
    if "pythia" in args.model_checkpoint:
        lm_weight = model.embed_out.weight.data.clone().detach()
    else:
        lm_weight = model.lm_head.weight.data
    with torch.no_grad():
        lm_weight_diff = torch.sum((old_lm_weight-lm_weight)**2,dim=1)

    for idx in range(len(data_source['input_ids'])):
        input_ids = np.array(data_source['input_ids'][idx])
        seq_len = np.sum(input_ids!=pad_token_id)
        cur_loss = 0

        for idx in range(seq_len-1, seq_len-1-args.mask_len, -1):
            token = input_ids[idx]
            cur_loss += lm_weight_diff[token].item()
        losses.append(cur_loss)
    torch.cuda.empty_cache()
    return losses

def evaluate_ppl(model, data_source, args, pad_token_id, cnt=-1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    losses1 = []
    losses2 = []    
    bsz = args.valid_batch_size
    for idx in range(int(np.ceil(len(data_source['input_ids'])/bsz))):
        torch.cuda.empty_cache()
        min_cnt = idx*bsz
        max_cnt = min((idx+1)*bsz,len(data_source['input_ids']))
        """
        if args.eval_full_loss:
            input_ids = torch.tensor(data_source['input_ids'][min_cnt:max_cnt])
            lm_labels = input_ids.detach().clone()
            lm_labels[lm_labels == pad_token_id] = -100
            input_ids, lm_labels = input_ids.to(args.device), lm_labels.to(args.device)
        else:
            input_ids, lm_labels = torch.tensor(data_source['input_ids'][min_cnt:max_cnt]).to(args.device), torch.tensor(data_source['labels'][min_cnt:max_cnt]).to(args.device)
        """
        input_ids = torch.tensor(data_source['input_ids'][min_cnt:max_cnt])
        lm_labels = input_ids.detach().clone()
        lm_labels[lm_labels == pad_token_id] = -100
        input_ids, lm_labels = input_ids.to(args.device), lm_labels.to(args.device)

        with torch.no_grad():
            output = model(input_ids)

            lm_logits = output[0]
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            lm_loss = criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
            lm_loss = lm_loss.reshape(input_ids.shape[0], -1)
            weight = torch.sum(lm_labels[..., 1:]!=-100, dim=1)
            loss = torch.sum(lm_loss, dim=1)/weight
            losses1.extend(loss.detach().cpu().numpy())

        input_ids, lm_labels = torch.tensor(data_source['input_ids'][min_cnt:max_cnt]).to(args.device), torch.tensor(data_source['labels'][min_cnt:max_cnt]).to(args.device)
        with torch.no_grad():
            output = model(input_ids)

            lm_logits = output[0]
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            lm_loss = criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
            lm_loss = lm_loss.reshape(input_ids.shape[0], -1)
            weight = torch.sum(lm_labels[..., 1:]!=-100, dim=1)
            loss = torch.sum(lm_loss, dim=1)/weight
            losses2.extend(loss.detach().cpu().numpy())            
            
            if cnt > 0 and len(losses1)>cnt:
                break
    torch.cuda.empty_cache()
    # aa=np.array(losses2)/np.array(losses1)
    # print(aa)
    # np.save(f"ratio_{args.eval_full_loss}.npy", aa)
    if args.eval_full_loss:
        return losses1
    else:
        return losses2
    # return losses

if __name__ == "__main__":
    train()
