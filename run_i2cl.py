import gc
import json
import copy
import time
import random
import argparse
import itertools
import torch
import numpy as np
from multiprocessing import Process, Queue

import utils
import my_datasets as md
import evaluator as ev

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
                                   max_data_num=None, seed=args.config['seed'])
    holdout_dataset = md.get_dataset(args.dataset_name, split='validation', 
                                     max_data_num=args.config['val_data_num'],
                                     sample_mode=args.config['sample_method'],
                                     seed=args.config['seed'])
    test_dataset = md.get_dataset(args.dataset_name, split='test', 
                                  max_data_num=args.config['test_data_num'],
                                  sample_mode=args.config['sample_method'],
                                  seed=args.config['seed'])

    # get max demonstration token length for each dataset
    if args.config['split_demon']:
        args.test_max_token = 1e8
    else:
        args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
        
    # get shot_num
    if args.dataset_name  == 'dbpedia':  # always use 1-shot for dbpedia
        args.config['shot_per_class'] = 1
        args.config['bs'] = 1
    args.shot_num = utils.get_shot_num(train_dataset, args.config['shot_per_class'])

    # build evaluators
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])
    holdout_evaluator = ev.Evaluator(holdout_dataset, batch_size=args.config['bs'])
    # init result_dict
    result_dict = {'demon': {},
                   'split_demon': {},
                   'test_result': {'zero_shot': [], 'few_shot': [], 'ours': []}, 
                   'linear_coef': {},
                   'time': {'calibrate': [], 'evaluate': []},
                   }
    cv_save_dict = {}
    
    for run_id in range(args.config['run_num']):
        run_name = f'run_{run_id}'
        args.run_name = run_name
        print(f'Run time {run_name}')
        run_seed = args.config['seed'] + run_id
        utils.set_seed(run_seed)
    
        # zero-shot baseline
        if run_id == 0 and args.config['run_baseline']:
            test_zeroshot_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='',
                                                           use_cache=args.config['use_cache'])
            result_dict['test_result']['zero_shot'].append(test_zeroshot_result)
            print(f'Test zero-shot result: {test_zeroshot_result}\n')

        # sample demonstration
        count = 0
        temp_demon_list, temp_result_list = [], []
        while True:
            demon, split_demon, demon_data_index = \
            train_dataset.gen_few_shot_demonstration(tokenizer=tokenizer, shot_num=args.shot_num, 
                                                     max_demonstration_tok_len=args.test_max_token,
                                                     add_extra_query=args.config['add_extra_query'],
                                                     example_separator=args.config['example_separator'],
                                                     gen_example_method = args.config['gen_example_method'],
                                                     return_data_index=True, seed=random.randint(0, 1e6))
            temp_demon_list.append((demon, split_demon, demon_data_index))
  #          import pdb;pdb.set_trace()
            if args.config['demo_sample_method'] == 'random':
                break
            else:
                tem_val_result = holdout_evaluator.evaluate(model_wrapper, tokenizer, 
                                                            demonstration=demon,
                                                            use_cache=args.config['use_cache'])
                temp_result = tem_val_result[args.metric]
                temp_result_list.append(temp_result)
            if count > 20:
                if args.config['demo_sample_method'] == 'deficient':
                    demon, split_demon, demon_data_index = temp_demon_list[np.argmin(temp_result_list)]
                else:
                    raise ValueError('Invalid demo_sample_method!')
                break
            count += 1

        # build val_evaluator use demon_data_index
        cali_dataset = copy.deepcopy(train_dataset)
        cali_dataset.all_data = [train_dataset.all_data[i] for i in demon_data_index]

        # clean demonstration
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
        
        # few-shot baseline
        if args.config['run_baseline']:
            test_fewshot_result = test_evaluator.evaluate(model_wrapper, tokenizer, 
                                                          demonstration=baseline_demon, 
                                                          use_cache=args.config['use_cache'])
            result_dict['test_result']['few_shot'].append(test_fewshot_result)
            print(f'Test few-shot result: {test_fewshot_result}\n')

        # generate demon_list
        # 根据数据集动态设置label2id
        if args.dataset_name == 'agnews':
            label2id = {
                "World": 0,
                "Sports": 1, 
                "Business": 2,
                "Technology": 3,
            }
        elif args.dataset_name == 'sst2':
            label2id = {
                "negative": 0,
                "positive": 1,
            }
        elif args.dataset_name == 'sst5':
            label2id = {
                "neutral": 0, 
                "great": 1, 
                "terrible": 2, 
                "positive": 3, 
                "negative": 4, 
            }
        elif args.dataset_name == 'trec':
            label2id = {
                "Abbreviation": 0,
                "Entity": 1,
                "Description": 2,
                "Person": 3,
                "Location": 4,
                "Number": 5,
            }
        elif args.dataset_name == 'mr':
            label2id = {
                "negative": 0,
                "positive": 1,
            }
        elif args.dataset_name == 'subj':
            label2id = {
                "objective": 0,
                "subjective": 1,
            }
        elif args.dataset_name == 'dbpedia':
            label2id = {
                "company": 0,
                "school": 1,
                "artist": 2,
                "athlete": 3,
                "politics": 4,
                "transportation": 5,
                "building": 6,
                "nature": 7,
                "village": 8,
                "animal": 9,
                "plant": 10,
                "album": 11,
                "film": 12,
                "book": 13,
            }
        elif args.dataset_name == 'hate_speech18':
            label2id = {
                "neutral": 0,
                "hate": 1,
            }
        elif args.dataset_name == 'emo':
            label2id = {
                "others": 0,
                "happy": 1,
                "sad": 2,
                "angry": 3,
            }
        else:
            # 默认使用SST2的标签
            label2id = {
                "negative": 0,
                "positive": 1,
            }
        label_id_list = []
        demon_list = [demon]
        split_demon_list = split_demon
        result_dict['demon'][run_name] = demon_list
        result_dict['split_demon'][run_name] = split_demon_list

        for d in split_demon_list:
            # 根据数据集使用不同的标签提取逻辑
            if args.dataset_name == 'agnews':
                # AGNews格式: "News: {text}\nType: {label}"
                if "Type:" in d:
                    label_text = d.split("Type:")[-1].strip().split("\n")[0].strip()
                    if label_text in label2id:
                        label_id_list.append(label2id[label_text])
            elif args.dataset_name == 'trec':
                # TREC格式: "Question: {text}\nAnswer Type: {label}"
                if "Answer Type:" in d:
                    label_text = d.split("Answer Type:")[-1].strip().split("\n")[0].strip()
                    if label_text in label2id:
                        label_id_list.append(label2id[label_text])
            else:
                # SST5格式: 标签词在最后一个空格之后
                if d.split(" ")[-1].split("\n")[0] in label2id:
                    label_id_list.append(label2id[d.split(" ")[-1].split("\n")[0]])
        
        # 添加调试信息
        print(f"Dataset: {args.dataset_name}")
        print(f"split_demon_list length: {len(split_demon_list)}")
        print(f"label_id_list length: {len(label_id_list)}")
        if len(split_demon_list) > 0:
            print(f"First demon: {repr(split_demon_list[0])}")
        
        assert len(label_id_list) == len(split_demon_list)

        # init strength_params
        model_wrapper.init_strength(args.config)

        # extract latents 
        all_latent_dicts = []
        with torch.no_grad():
            if not args.config['split_demon']:
                target_demon_list = demon_list[0]
            else:
                target_demon_list = split_demon_list
            for cur_demon in target_demon_list:
                with model_wrapper.extract_latent():
                    demon_token = tokenizer(cur_demon, return_tensors='pt').to(args.device)
                    # import pdb; pdb.set_trace()
                    _ = model(**demon_token)
                all_latent_dicts.append(model_wrapper.latent_dict)
                model_wrapper.reset_latent_dict()

        # generate context vector 
        result = model_wrapper.get_context_vector(all_latent_dicts, args.config)
       # import pdb;pdb.set_trace()
        
        # 检查是否返回了聚类结果（kmeans/hier模式）
        if isinstance(result, list) and len(result) == 2:
            # kmeans/hier模式：result = [context_vector_dict, cluster_dict]
            context_vector_dict, cluster_dict = result
            # 保存聚类结果
            cluster_save_dict = {}
            for layer, subdict in cluster_dict.items():
                cluster_save_dict[layer] = {}
                for module, cluster_result in subdict.items():
                    cluster_save_dict[layer][module] = {
                        'cluster': cluster_result['cluster'].cpu().numpy().tolist(),
                        'index': cluster_result['index'],
                        'mean_purity': utils.mean_purity(cluster_result['index'], label_id_list)
                    }
            with open(args.save_dir + '/cluster_dict.json', 'w') as f:
                json.dump(cluster_save_dict, f, indent=4)
            # 将聚类信息添加到context_vector_dict中，以便传递给inject_latent
            context_vector_dict = [context_vector_dict, cluster_dict]
        else:
            # 正常模式：result = context_vector_dict
            context_vector_dict = result
            
        if args.config['gen_cv_method'] == 'noise':
            context_vector_dict = model_wrapper.init_noise_context_vector(context_vector_dict)
        del all_latent_dicts
            
        # calibrate context vector (skip for static_add)
        if args.config['inject_method'] == 'static_add':
            print('Static_add: skipping calibration, direct injection only.')
            s_t = time.time()
            # 只初始化strength参数，不进行训练
            model_wrapper.init_strength(args.config)
            e_t = time.time()
            print(f'Static_add setup time: {e_t - s_t}')
            result_dict['time']['calibrate'].append(e_t - s_t)
        else:
            s_t = time.time()
            model_wrapper.calibrate_strength(result, cali_dataset, 
                                             args.config, save_dir=args.save_dir, 
                                             run_name=args.run_name)
            e_t = time.time()
            print(f'Calibration time: {e_t - s_t}')
            result_dict['time']['calibrate'].append(e_t - s_t)

        # save linear_coef (static_add will be None)
        if model_wrapper.linear_coef is not None:
            result_dict['linear_coef'][run_name] = model_wrapper.linear_coef.tolist()
        else:
            result_dict['linear_coef'][run_name] = None

        # evaluate i2cl
        s_t = time.time()
        with torch.no_grad():
            with model_wrapper.inject_latent(context_vector_dict, args.config,
                                             model_wrapper.linear_coef):
                test_ours_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='', use_cache=args.config['use_cache'])
                print(f'Test I2CL result: {test_ours_result}\n')
                result_dict['test_result']['ours'].append(test_ours_result)
        e_t = time.time()
        print(f'Evaluate time: {e_t - s_t}')
        result_dict['time']['evaluate'].append(e_t - s_t)

        # save result_dict after each run
        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(result_dict, f, indent=4)

        # save context vector dict
        if args.config['post_fuse_method'] in ['kmeans', 'hier', 'sim']:
            # kmeans/hier模式：context_vector_dict是列表 [context_vector_dict, cluster_dict]
            if isinstance(context_vector_dict, list):
                # 只保存第一个元素（context_vector_dict）
                save_dict = context_vector_dict[0]
            else:
                save_dict = context_vector_dict
            # 保存context vector
            for layer, subdict in save_dict.items():
                for module, activation in subdict.items():
                    save_dict[layer][module] = activation.cpu().numpy().tolist()
            cv_save_dict[run_name] = save_dict
        else:
            # 正常模式
            for layer, subdict in context_vector_dict.items():
                for module, activation in subdict.items():
                    context_vector_dict[layer][module] = activation.cpu().numpy().tolist()
            cv_save_dict[run_name] = context_vector_dict

        with open(args.save_dir + '/cv_save_dict.json', 'w') as f:
            json.dump(cv_save_dict, f, indent=4)

    # delete all variables
    del model_wrapper, model, tokenizer, train_dataset, cali_dataset, test_dataset, holdout_dataset
    del test_evaluator, holdout_evaluator
    del result_dict, context_vector_dict, demon_list
            

# get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_i2cl.py', help='path to config file')
    parser.add_argument('--datasets', nargs='+', default=None, help='dataset name')
    parser.add_argument('--inject_method', type=str, default=None, help='inject method')
    parser.add_argument('--post_fuse_method', type=str, default=None, help='post fuse method')
    parser.add_argument('--svd_topk', type=int, default=1, help='svd topk')
    parser.add_argument('--kmeans_n_clusters', type=int, default=5, help='kmeans n clusters')
    parser.add_argument('--kmeans_random_state', type=int, default=42, help='kmeans random state')
    parser.add_argument('--query_pool_method', type=str, default='mean', help='query pool method')
    parser.add_argument('--shot_per_class', type=int, default=5, help='shot per class')
    parser.add_argument('--tok_pos', type=str, default='label', help='token position')

    return parser.parse_args()


if __name__ == "__main__":
    # get args
    args = get_args()
    # load config
    config = utils.load_config(args.config_path)
    # Generate all combinations of models and datasets
    # 自动生成exp_name格式: datasets_post_fuse_method_kmeans_n_clusters_inject_method_tok_pos_shot_per_class
    exp_name_parts = [
        args.datasets[0] if isinstance(args.datasets, list) else args.datasets,  # 取第一个数据集名称
        args.post_fuse_method,
        str(args.kmeans_n_clusters),
        args.inject_method,
        args.tok_pos,
        str(args.shot_per_class)
    ]
    config['exp_name'] = 'exps/'+'_'.join(exp_name_parts) + '_mid'
    if args.datasets is not None:
        config['datasets'] = args.datasets
    if args.inject_method is not None:
        config['inject_method'] = args.inject_method
    if args.post_fuse_method is not None:
        config['post_fuse_method'] = args.post_fuse_method
    if args.svd_topk is not None:
        config['svd_topk'] = args.svd_topk
    if args.kmeans_n_clusters is not None:
        config['kmeans_n_clusters'] = args.kmeans_n_clusters
    if args.kmeans_random_state is not None:
        config['kmeans_random_state'] = args.kmeans_random_state
    if args.query_pool_method is not None:
        config['query_pool_method'] = args.query_pool_method
    if args.shot_per_class is not None:
        config['shot_per_class'] = args.shot_per_class
    if args.tok_pos is not None:
        config['tok_pos'] = args.tok_pos
    combinations = list(itertools.product(config['models'], config['datasets']))
    # Queue to hold tasks
    task_queue = Queue()
    for combine in combinations:
        task_queue.put(combine)

    def run_task(gpu_id, config):
        while not task_queue.empty():
            model_name, dataset_name = task_queue.get()
            print(f"Running {model_name} on {dataset_name} with GPU {gpu_id}")
            input_args = argparse.Namespace()
            cur_config = copy.deepcopy(config)
            input_args.model_name = model_name
            input_args.dataset_name = dataset_name
            input_args.gpu = gpu_id
            input_args.config = cur_config
            try:
                main(input_args)
            finally:
                # Clean up CUDA memory after each task
                gc.collect()
                torch.cuda.empty_cache()
                print(f"CUDA memory cleared for GPU {gpu_id}") 
                time.sleep(5)

    # Create a process for each GPU
    processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
    # Start all processes
    for p in processes:
        p.start()
    # Wait for all processes to finish
    for p in processes:
        p.join()
    print("All tasks completed.")
# if __name__ == "__main__":
#     # get args
#     args = get_args()
#     config = utils.load_config(args.config_path)
#     if args.exp_name is not None:
#         config['exp_name'] = args.exp_name
#     if args.models is not None:
#         config['models'] = args.models
#     if args.datasets is not None:
#         config['datasets'] = args.datasets
#     if args.bs is not None:
#         config['bs'] = args.bs
#     if args.inject_method is not None:
#         config['inject_method'] = args.inject_method
#     if args.init_value is not None:
#         config['init_value'] = args.init_value
#     if args.post_fuse_method is not None:
#         config['post_fuse_method'] = args.post_fuse_method
#     if args.svd_topk is not None:
#         config['svd_topk'] = args.svd_topk
#     if args.kmeans_n_clusters is not None:
#         config['kmeans_n_clusters'] = args.kmeans_n_clusters
#     if args.kmeans_random_state is not None:
#         config['kmeans_random_state'] = args.kmeans_random_state
#     if args.query_pool_method is not None:
#         config['query_pool_method'] = args.query_pool_method
#     if args.shot_per_class is not None:
#         config['shot_per_class'] = args.shot_per_class
#     model_name, dataset_name = config['models'][0], config['datasets'][0]

#     input_args = argparse.Namespace()
#     input_args.model_name = model_name
#     input_args.dataset_name = dataset_name
#     input_args.gpu = config['gpus'][0]
#     input_args.config = copy.deepcopy(config)
#     import pdb; pdb.set_trace()

#     main(input_args)  # 单进程直接跑，pdb 就能用