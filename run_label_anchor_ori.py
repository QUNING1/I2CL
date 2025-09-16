import gc
import json
import copy
import time
import random
import argparse
import itertools
import torch
import torch.nn.functional as F
from multiprocessing import Process, Queue

import utils
import my_datasets as md
import evaluator as ev


def merge_and_replace_past_key_values(past_key_values, attn_mask, model_name, merge_method='mean'):
    # 确保attn_mask在正确的设备上
    device = attn_mask.device
    # 找到所有为True的位置索引
    true_positions = torch.where(attn_mask.squeeze(0))[0]
    if len(true_positions) == 0:
        # 如果没有True的位置，返回原始数据
        return past_key_values
    
    # 最后一个True的位置
    last_true_idx = true_positions[-1]
    # 处理past_key_values
    new_past_key_values = []
    for layer_idx, layer in enumerate(past_key_values):
        new_layer = []
        for kv_idx, tensor in enumerate(layer):  # 0: key, 1: value
            # 确保tensor在正确的设备上
            tensor = tensor.to(device)
            
            # 对所有True位置的张量取平均值
            if 'llama' in model_name.lower():
                # Llama模型的处理方式 - 按索引切片
                true_tensors = tensor[:, :, true_positions, :]
            else:
                # 其他模型的处理方式 - 使用布尔掩码
                true_tensors = tensor[:, :, attn_mask.squeeze(0), :]
            
            # merge
            if merge_method == 'mean':
                merged_tensor = torch.mean(true_tensors, dim=2, keepdim=True)
            else:
                raise ValueError(f'merge_method: {merge_method} not supported')
            
            # 创建新的张量，只有最后一个位置被替换为合并后的值
            new_tensor = tensor.clone()
            if 'llama' in model_name.lower():
                # Llama模型 - 直接替换最后一个True位置
                new_tensor[:, :, 0:1, :] = merged_tensor.to(device)
            else:
                # 其他模型 - 由于已经使用掩码过滤，这里需要特殊处理
                # 找到在原tensor中的对应位置
                for i, pos in enumerate(true_positions):
                    if i == len(true_positions) - 1:  # 最后一个位置
                        new_tensor[:, :, pos:pos+1, :] = merged_tensor.to(device)
            
            new_layer.append(new_tensor)
        new_past_key_values.append(tuple(new_layer))
    
    # 确保返回的所有张量都在正确的设备上
    return tuple(new_past_key_values)

def cached_evaluation(config, dataset, model, tokenizer, compressed_past_key_values, 
                      compressed_attn_mask, position_offset=None):
    batch_size = config['bs']
    # prepare label dict          
    label_map = {}
    ans_txt_list = dataset.get_dmonstration_template()['options']
    for label, ans_txt in enumerate(ans_txt_list):
        if 'gpt' in tokenizer.__class__.__name__.lower():
            ans_txt = ' ' + ans_txt  # add space to the beginning of answer
        ans_tok = tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
        print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
        label_map[ans_tok] = label  # index is the label
    print(f"label_map: {label_map}")

    # prepare all data
    all_pred_labels = []
    all_inputs, all_labels = [], []
    for data in dataset.all_data:
        ques_str, _, label = dataset.apply_template(data)
        context = ques_str
        all_inputs.append(context)
        all_labels.append(label)
    # prepare cached data
    cached_past_key_values = tuple(tuple(t.repeat(batch_size, 1, 1, 1) for t in tup) 
                                   for tup in compressed_past_key_values)
    cached_attn_mask = compressed_attn_mask.repeat(batch_size, 1).to(model.device)

    # loop over all data
    with torch.no_grad():
        for i in range(0, len(all_inputs), batch_size):
            cur_inputs = all_inputs[i:i+batch_size]
            input_tok = tokenizer(cur_inputs, return_tensors="pt", padding=True)
            input_ids = input_tok['input_ids'].to(model.device)
            attn_mask = input_tok['attention_mask'].to(model.device)

            # get index for prediction logits, need to be applied before concatenating demon_attn_mask with attn_mask
            pred_loc = utils.last_one_indices(attn_mask).to(model.device)
            # get logits
            if position_offset is not None:
                position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=model.device).repeat(input_ids.size(0), 1)
                position_ids = position_ids + position_offset
            else:
                position_ids = None

            attn_mask = torch.cat([cached_attn_mask, attn_mask], dim=1)
            #import pdb; pdb.set_trace()
            output = model(input_ids=input_ids, attention_mask=attn_mask,
                           past_key_values=cached_past_key_values,
                           position_ids=position_ids)
            # get prediction logits
            logits = output.logits
            pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
            # get prediction labels
            interest_index = list(label_map.keys())
            pred_logits = pred_logits[:, interest_index]
            probs = F.softmax(pred_logits, dim=-1)
            pred_labels = probs.argmax(dim=-1)
            # save results
            all_pred_labels.extend(pred_labels.cpu().numpy().tolist())

        assert len(all_pred_labels) == len(all_labels)
        # both all_results and all_labels are list containing label index, can you help me to calculate accuracy and macro f1?
        # initialize TP, FP, FN
        acc = []
        num_classes = dataset.class_num
        TP = [0] * num_classes
        FP = [0] * num_classes
        FN = [0] * num_classes
        for i, true_label in enumerate(all_labels):
            pred_label = all_pred_labels[i]
            pred = pred_label == true_label
            acc.append(pred)
            # Update TP, FP, FN
            if pred:
                TP[true_label] += 1
            else:
                FP[pred_label] += 1
                FN[true_label] += 1
        # Calculate precision, recall, F1 for each class and macro F1
        precision = [0] * num_classes
        recall = [0] * num_classes
        f1 = [0] * num_classes
        for i in range(num_classes):
            precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
            recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        macro_f1 = sum(f1) / num_classes
        acc = sum(acc) / len(acc)
        return {'acc': acc, 'macro_f1': macro_f1}
    

def main(args):
    # set global seed
    utils.set_seed(args.config['seed'])
    # set device
    args.device = utils.set_device(args.gpu)
    # set metric used
    args.metric = args.config['metric']
    # get save dir
    utils.init_exp_path(args, args.config['exp_name'])

    # load tokenizer and model
    model, tokenizer, model_config = \
    utils.load_model_tokenizer(args.model_name, args.device)

    # get model_wrapper
    model_wrapper = utils.get_model_wrapper(args.model_name, model, 
                                            tokenizer, model_config, 
                                            args.device)
    # load datasets
    train_dataset = md.get_dataset(args.dataset_name, split='train', 
                                   max_data_num=None,
                                   seed=args.config['seed'])
    test_dataset = md.get_dataset(args.dataset_name, split='test', 
                                  max_data_num=args.config['test_data_num'],
                                  sample_mode=args.config['sample_method'], 
                                  seed=args.config['seed'])

    # get max demonstration token length for each dataset
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
    
    # get shot_num
    if args.dataset_name  == 'dbpedia':  # always use 1-shot for dbpedia
        args.config['shot_per_class'] = 1
        args.config['bs'] = 1
    args.shot_num = utils.get_shot_num(train_dataset, args.config['shot_per_class'])
    # build evaluate
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])
    # init result_dict
    result_dict = {'demon': {},
                   'split_demon': {},
                   'test_result': {'zero_shot': [], 'few_shot': [], 'ours': []}, 
                   'time': {'calibrate': [], 'evaluate': []},
                   }
    
    for run_id in range(args.config['run_num']):
        run_name = f'run_{run_id}'
        args.run_name = run_name
        print(f'Run time {run_name}')
        run_seed = args.config['seed'] + run_id
        utils.set_seed(run_seed)

        # # zero-shot baseline
        # if run_id == 0 and args.config['run_baseline']:  
        #     test_zeroshot_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='',
        #                                                    use_cache=args.config['use_cache'])
        #     result_dict['test_result']['zero_shot'].append(test_zeroshot_result)
        #     print(f'Test zero-shot result: {test_zeroshot_result}\n')

        # sample demonstration
        demon, _ = \
        train_dataset.gen_few_shot_demonstration(tokenizer=tokenizer, shot_num=args.shot_num, 
                                                max_demonstration_tok_len=args.test_max_token,
                                                add_extra_query=args.config['add_extra_query'],
                                                example_separator=args.config['example_separator'],
                                                seed=random.randint(0, 1e6))

        if args.config['add_extra_query']:
            first_format_anchor = train_dataset.get_dmonstration_template()['format'][0]
            # remove all contents after the last first_format_anchor including the anchor
            if first_format_anchor in demon:
                baseline_demon = demon[:demon.rfind(first_format_anchor)]
                query_demon = demon[demon.rfind(first_format_anchor):]
        else:
            baseline_demon = demon
            query_demon = None
        print(f'Demonstration:\n{demon}\n')
        print(f'Baseline demonstration:\n{baseline_demon}\n')
        print(f'Query demonstration:\n{query_demon}\n')
        
        # # few-shot baseline
        # if args.config['run_baseline']:
        #     test_fewshot_result = test_evaluator.evaluate(model_wrapper, tokenizer, 
        #                                                  demonstration=baseline_demon, 
        #                                                  use_cache=args.config['use_cache'])
        #     result_dict['test_result']['few_shot'].append(test_fewshot_result)
        #     print(f'Test few-shot result: {test_fewshot_result}\n')

        # apply label anchor
        list_label = None
        if args.config['merge_label']:
            if args.dataset_name == 'agnews':
                list_label = ['World', 'Sports', 'Business', 'Technology']
            elif args.dataset_name == 'sst5':
                list_label = ["terrible", "negative", "neutral", "positive", "great"]
            elif args.dataset_name == 'sst2':
                list_label = ['negative', 'positive']
            elif args.dataset_name == 'dbpedia':
                list_label = ['company', 'school', 'artist', 'athlete', 
                        'politics', 'transportation', 'building', 
                        'nature', 'village', 'animal', 'plant',
                        'album', 'film', 'book']
            elif args.dataset_name == 'trec':
                list_label = ["Abbreviation", "Entity", "Description", "Person", "Location", "Number"]
            elif args.dataset_name == 'hate_speech18':
                list_label = ['neutral', 'hate']
            else:
                raise ValueError(f'dataset_name: {args.dataset_name} not supported')
        context_solver = utils.ContextSolver(task_name=args.dataset_name, tokenizer=tokenizer)
        demon_token = tokenizer(demon, return_tensors='pt').to(args.device)
        compress_attn_mask = context_solver.get_mask(input_ids=demon_token['input_ids'], list_label=list_label).to(args.device)
        print(f'Compress_attn_mask: {compress_attn_mask}\n')
        # compressed_text
        compressed_token_id = copy.deepcopy(demon_token['input_ids'][0])
        mask = copy.deepcopy(compress_attn_mask).cpu().detach().numpy()
        compressed_token_id = compressed_token_id.cpu().detach().numpy()
        compressed_text = tokenizer.decode(list(compressed_token_id[mask]))
        print(f'Compressed_text: {compressed_text}\n')
        # # 检查compressed_text是否包含所有demons
        # if list_label is not None:
        #     assert torch.where(compress_attn_mask)[0].shape[0] == args.config['shot_per_class'] * len(list_label)

        with torch.no_grad():
            demon_outputs = model(**demon_token, use_cache=True)
        past_key_values = demon_outputs.past_key_values

        # if args.model_name == 'meta-llama/Llama-2-7b-hf': 
        if 'llama' in args.model_name.lower():
            mask_end_idx = torch.where(compress_attn_mask)[0][-1] + 1
            cached_past_key_values = tuple(tuple(t[:, :, :mask_end_idx, :] for t in tup) for tup in past_key_values)
            cached_attn_mask = copy.deepcopy(compress_attn_mask)[:mask_end_idx].unsqueeze(0)
        else:
            cached_past_key_values =  tuple(tuple(t[:, :, compress_attn_mask, :] for t in tup) for tup in past_key_values)
            cached_attn_mask = torch.ones(1, compress_attn_mask.sum(), dtype=torch.bool).to(args.device)
        print(f'Cached_attn_mask: {cached_attn_mask}\n')
        
        if args.model_name == 'gpt2-xl':
            position_offset = 0
        elif args.model_name == 'EleutherAI/gpt-j-6B':
            position_offset = torch.where(compress_attn_mask)[0][-1] + 1
        # elif args.model_name == 'meta-llama/Llama-2-7b-hf':
        elif 'llama' in args.model_name.lower():
            position_offset = None
        else:
            raise ValueError('model not supported')
        
        # 处理cached_past_key_values和cached_attn_mask，合并所有label到最后一个
        if args.config['merge_label']:
            cached_past_key_values = merge_and_replace_past_key_values(cached_past_key_values, cached_attn_mask, args.model_name, args.config['merge_method'])
            # 修改cached_attn_mask,只保留最后一个True，其余位置保持为False
           # import pdb;pdb.set_trace()
            cached_attn_mask = torch.zeros_like(cached_attn_mask, dtype=torch.bool, device=args.device)
            cached_attn_mask[:, 0] = True
        
        # evaluate with label anchor
        s_t = time.time()
        test_result = cached_evaluation(args.config, test_dataset, model, tokenizer,
                                        cached_past_key_values, cached_attn_mask, position_offset)
        print(f'Test label_anchor result: {test_result}\n')
        result_dict['test_result']['ours'].append(test_result)
        e_t = time.time()
        result_dict['time']['evaluate'].append(e_t - s_t)

        # save result_dict after each run
        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(result_dict, f, indent=4)
    
    # delete all variables
    del model_wrapper, model, tokenizer, train_dataset, test_dataset
    del test_evaluator
    del result_dict
            
# get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_label_anchor.py', help='path to config file')
    return parser.parse_args()

# if __name__ == "__main__":
#     # get args
#     args = get_args()
#     # load config
#     config = utils.load_config(args.config_path)
#     # Generate all combinations of models and datasets
#     combinations = list(itertools.product(config['models'], config['datasets']))
#     # Queue to hold tasks
#     task_queue = Queue()
#     for combine in combinations:
#         task_queue.put(combine)

#     def run_task(gpu_id, config):
#         while not task_queue.empty():
#             model_name, dataset_name = task_queue.get()
#             print(f"Running {model_name} on {dataset_name} with GPU {gpu_id}")
#             input_args = argparse.Namespace()
#             cur_config = copy.deepcopy(config)
#             input_args.model_name = model_name
#             input_args.dataset_name = dataset_name
#             input_args.gpu = gpu_id
#             input_args.config = cur_config
#             try:
#                 main(input_args)
#             finally:
#                 # Clean up CUDA memory after each task
#                 gc.collect()
#                 torch.cuda.empty_cache()
#                 print(f"CUDA memory cleared for GPU {gpu_id}") 
#                 time.sleep(5)

#     # Create a process for each GPU
#     processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
#     # Start all processes
#     for p in processes:
#         p.start()
#     # Wait for all processes to finish
#     for p in processes:
#         p.join()
#     print("All tasks completed.")

if __name__ == "__main__":
    # get args
    args = get_args()
    config = utils.load_config(args.config_path)
    model_name, dataset_name = config['models'][0], config['datasets'][0]

    input_args = argparse.Namespace()
    input_args.model_name = model_name
    input_args.dataset_name = dataset_name
    input_args.gpu = config['gpus'][0]
    input_args.config = copy.deepcopy(config)
    main(input_args)  # 单进程直接跑，pdb 就能用