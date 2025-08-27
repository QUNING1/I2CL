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
    utils.set_seed(args.config['seed'])
    args.device = utils.set_device(args.gpu)
    args.metric = args.config['metric']
    utils.init_exp_path(args, args.config['exp_name'])

    model, tokenizer, model_config = utils.load_model_tokenizer(args.model_name, args.device)
    model_wrapper = utils.get_model_wrapper(args.model_name, model, tokenizer, model_config, args.device)

    train_dataset = md.get_dataset(args.dataset_name, split='train', max_data_num=None, seed=args.config['seed'])
    test_dataset = md.get_dataset(args.dataset_name, split='test', max_data_num=args.config['test_data_num'], sample_mode=args.config['sample_method'], seed=args.config['seed'])

    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)

    if args.dataset_name == 'dbpedia':
        args.config['shot_per_class'] = 1
        args.config['bs'] = 1
    args.shot_num = utils.get_shot_num(train_dataset, args.config['shot_per_class'])

    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])
    result_dict = {
        'demon': {},
        'test_result': {'zero_shot': [], 'few_shot': []},
        'time': {'evaluate': []},
    }

    for run_id in range(args.config['run_num']):
        run_name = f'run_{run_id}'
        args.run_name = run_name
        print(f'Run time {run_name}')
        utils.set_seed(args.config['seed'] + run_id)

        # zero-shot
        if run_id == 0 and args.config['run_baseline']:
            zero_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='', use_cache=args.config['use_cache'])
            result_dict['test_result']['zero_shot'].append(zero_result)
            print(f'Test zero-shot result: {zero_result}\n')

        # sample few-shot demonstration
        demon, split_demon = train_dataset.gen_few_shot_demonstration(
            tokenizer=tokenizer,
            shot_num=args.shot_num,
            max_demonstration_tok_len=args.test_max_token,
            add_extra_query=args.config['add_extra_query'],
            example_separator=args.config['example_separator'],
            gen_example_method=args.config['gen_example_method'],
            return_data_index=False,
            seed=random.randint(0, 1e6)
        )

        print(f'Few-shot demonstration:\n{demon}\n')
        result_dict['demon'][run_name] = demon

        # evaluate few-shot
        s_t = time.time()
        few_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration=demon, use_cache=args.config['use_cache'])
        e_t = time.time()
        print(f'Test few-shot result: {few_result}\n')

        result_dict['test_result']['few_shot'].append(few_result)
        result_dict['time']['evaluate'].append(e_t - s_t)

        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(result_dict, f, indent=4)

    del model_wrapper, model, tokenizer, train_dataset, test_dataset, test_evaluator, result_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_icl.py')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config = utils.load_config(args.config_path)
    combinations = list(itertools.product(config['models'], config['datasets']))
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
                gc.collect()
                torch.cuda.empty_cache()
                print(f"CUDA memory cleared for GPU {gpu_id}")
                time.sleep(5)

    processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("All tasks completed.")
