import math
import string
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from functools import reduce
import numpy as np
import utils
import global_vars as gv
from peft import get_peft_model, PromptTuningConfig


class ModelWrapper(nn.Module):                                                                                                                                                            
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device
        self.num_layers = self._get_layer_num()
        self.latent_dict = {}
        self.linear_coef = None
        self.inject_layers = None
        print(f"The model has {self.num_layers} layers:")

    def reset_latent_dict(self):
        self.latent_dict = {}
            
    @contextmanager
    def extract_latent(self):
        handles = []
        try:
            # attach hook
            for layer_idx in range(self.num_layers):
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'attn')).register_forward_hook(
                    self.extract_hook_func(layer_idx, 'attn')))
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'mlp')).register_forward_hook(
                    self.extract_hook_func(layer_idx, 'mlp')))
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'hidden')).register_forward_hook(
                    self.extract_hook_func(layer_idx, 'hidden')))
            yield
        finally:
            # remove hook
            for handle in handles:
                handle.remove()

    def extract_hook_func(self, layer_idx, target_module):
        if layer_idx not in self.latent_dict:
            self.latent_dict[layer_idx] = {}
        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                outputs = outputs[0]
            self.latent_dict[layer_idx][target_module] = outputs.detach().cpu()
        return hook_func
    
    @contextmanager
    def inject_latent(self, context_vector_dict, config, linear_coef, train_mode=False):
        handles = []
        assert self.inject_layers is not None, "inject_layers is not set!"
        inject_method = config['inject_method']
        inject_pos = config['inject_pos']
        add_noise = config['add_noise']
        noise_scale = config['noise_scale']
        cluster_info, cluster_index = None, None
        is_using_cluster = config['is_using_cluster']
        try:
            # attach hook
            for layer_idx, layer in enumerate(self.inject_layers):
                for module_idx, module in enumerate(config['module']):
                    # 检查context_vector_dict是否为列表（kmeans模式）
                    if isinstance(context_vector_dict, list) and len(context_vector_dict) == 2:
                        # 聚类模式：context_vector_dict[0][layer][module] 是 (num_examples, dim)
                        # 我们需要取所有example的向量用于聚类
                        all_example_vectors = context_vector_dict[0][layer][module].to(self.device)  # (num_examples, dim)
                        context_vector_container = [all_example_vectors]
                        cluster_info = context_vector_dict[1][layer][module]['cluster'].to(self.device)
                        cluster_index = context_vector_dict[1][layer][module]['index']
                    else:
                        # 普通模式：context_vector_dict是字典，没有聚类信息
                        context_vector_container = [context_vector_dict[layer][module].to(self.device)]
                        cluster_info = None
                        cluster_index = None
                    
                    # static_add方法不需要强度系数
                    if inject_method == 'static_add':
                        strength = None
                    else:
                        strength = linear_coef[layer_idx, module_idx, :]
                    
                    inject_func = self.inject_hook_func(context_vector_container, strength,
                                                        inject_method, add_noise, noise_scale, 
                                                        inject_pos, train_mode, is_using_cluster, cluster_info, cluster_index, config)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func)
                        )
            yield
        finally:
            # remove hook
            print(f"Removing {len(handles)} hooks...")
            for handle in handles:
                handle.remove()

    def inject_hook_func(self, context_vector_container, strength, inject_method,
                         add_noise, noise_scale, inject_pos, train_mode=False, is_using_cluster=False, cluster_info=None, cluster_index=None, config=None):

        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                output = outputs[0]     
            else:
                output = outputs
            # init context_vector
            # import pdb; pdb.set_trace()
            context_vector = context_vector_container[0]
            
            # 检查是否为kmeans模式（context_vector形状为(num_examples, dim)）
            if context_vector.dim() == 2 and cluster_info is not None:
                # 聚类/相似度模式：使用动态计算的context vector
                context_vector = self._get_cluster_context_vector(output, context_vector, cluster_info, cluster_index, is_using_cluster, config)
            else:
                # 普通模式：expand inject_value to match output size (b, seq_len, d)
                context_vector = context_vector.expand(output.size(0), output.size(1), context_vector.size(-1))
            
            if inject_method == 'add':
                output = output + F.relu(strength) * context_vector
            elif inject_method == 'static_add':
                output = output + 0.1 * context_vector
            elif inject_method == 'linear':
                if inject_pos == 'all':
                    output = strength[1] * output + strength[0] * context_vector
                else:
                    if inject_pos == 'last':
                        for i in range(output.size(0)):
                            end_idx = gv.ATTN_MASK_END[i] - 1
                            content = strength[1] * output[i, end_idx, :].clone().detach() + strength[0] * context_vector[i, end_idx, :]
                            output[i, end_idx, :] = content
                    elif inject_pos == 'first':
                        content = strength[1] * output[:, 0, :].clone().detach() + strength[0] * context_vector[:, 0, :]
                        output[:, 0, :] = content
                    elif inject_pos == 'random':
                        for i in range(output.size(0)):
                            end_idx = gv.ATTN_MASK_END[i]
                            random_idx = random.randint(0, end_idx)
                            content = strength[1] * output[i, random_idx, :].clone().detach() + strength[0] * context_vector[i, random_idx, :]
                            output[i, random_idx, :] = content
                    else:
                        raise ValueError("only support all, last, first or random!")
                    
            elif inject_method == 'balance':
                a, b = strength[0], strength[1]
                output = ((1.0 - a) * output + a * context_vector) * b
            else:
                raise ValueError("only support add, linear, balance or static_add!")

            if add_noise and train_mode:
                # get l2_norm of output and use it as a scalar to scale noise, make sure no gradient
                output_norm = torch.norm(output, p=2, dim=-1).detach().unsqueeze(-1)
                noise = torch.randn_like(output).detach()
                output += noise * output_norm * noise_scale
            
            if type(outputs) is tuple:
                outputs = list(outputs)
                outputs[0] = output
                outputs = tuple(outputs)
            else:
                outputs = output
            return outputs
        return hook_func
    

    @contextmanager
    def replace_latent(self, context_vector_dict, target_layers, config):
        handles = []
        try:
            # attach hook
            for _, layer in enumerate(target_layers):
                for _, module in enumerate(config['module']):
                    context_vector_container = [context_vector_dict[layer][module].to(self.device)]
                    inject_func = self.replace_hook_func(context_vector_container)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func))
            yield
        finally:
            # remove hook
            print(f"Removing {len(handles)} hooks...")
            for handle in handles:
                handle.remove()

    def replace_hook_func(self, context_vector_container):
        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                output = outputs[0]     
            else:
                output = outputs
            # init context_vector
            context_vector = context_vector_container[0]
            # replace hidden states of last token position with context_vector
            for i in range(output.size(0)):
                end_idx = gv.ATTN_MASK_END[i]
                output[i, end_idx, :] = context_vector
            
            if type(outputs) is tuple:
                outputs = list(outputs)
                outputs[0] = output
                outputs = tuple(outputs)
            else:
                outputs = output
            return outputs
        return hook_func
    

    def get_context_vector(self, all_latent_dicts, config):
        if len(all_latent_dicts) == 1:
            latent_dict = all_latent_dicts[0]
            output_dict = {}
            for layer, sub_dict in latent_dict.items():
                output_dict[layer] = {}
                for module in config['module']:
                    latent_value = sub_dict[module]
                    if config['tok_pos'] == 'last':
                        latent_value = latent_value[:, -1, :].squeeze()
                    elif config['tok_pos'] == 'first':
                        latent_value = latent_value[:, 0, :].squeeze()
                    elif config['tok_pos'] == 'random':
                        latent_value = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                    else:
                        raise ValueError("only support last, first or random!")
                    output_dict[layer][module] = latent_value.detach().to('cpu')
        else:
            # concatenate context vector for each module
            ensemble_dict = {module:[] for module in config['module']} # {module_name: []}
            for _, latent_dict in enumerate(all_latent_dicts):
                cur_dict = {module:[] for module in config['module']}  # {module_name: []}
                for layer, sub_dict in latent_dict.items():
                    for module in config['module']:
                        latent_value = sub_dict[module]  # (b, seq_len, d)  
                        if config['tok_pos'] == 'last':
                            latent_value = latent_value[:, -1, :].squeeze()
                        elif config['tok_pos'] == 'label':
                            latent_value = latent_value[:, -2, :].squeeze()  # 取倒数第二个token（标签词）
                        elif config['tok_pos'] == 'first':
                            latent_value = latent_value[:, 0, :].squeeze()
                        elif config['tok_pos'] == 'random':
                            latent_value = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                        else:
                            raise ValueError("only support last, label, first or random!")
                        cur_dict[module].append(latent_value)

                for module, latent_list in cur_dict.items():
                    cur_latent = torch.stack(latent_list, dim=0) # (layer_num, d)
                    ensemble_dict[module].append(cur_latent)
            for module, latent_list in ensemble_dict.items():
                if config['post_fuse_method'] == 'mean':
                    context_vector = torch.stack(latent_list, dim=0).mean(dim=0)  # (layer_num, d)
                    ensemble_dict[module] = context_vector 
                elif config['post_fuse_method'] == 'pca':
                    latents = torch.stack(latent_list, dim=0)  # (ensemble_num, layer_num, d)
                    ensemble_num, layer_num, d = latents.size()
                    latents = latents.view(ensemble_num, -1)  # (ensemble_num*layer_num, d)
                    # apply pca
                    pca = utils.PCA(n_components=1).to(latents.device).fit(latents.float())
                    context_vector = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0)
                    ensemble_dict[module] = context_vector.view(layer_num, d)  # (layer_num, d)
                elif config['post_fuse_method'] == 'svd':
                    # 简单的SVD加权融合（支持前K个方向）
                    svd_topk = config.get('svd_topk', None)
                    context_vector = self._simple_svd_fusion(latent_list, topk=svd_topk)
                    ensemble_dict[module] = context_vector
                elif config['post_fuse_method'] == 'kmeans':
                    # kmeans聚类方法，返回聚类结果
                    return self._get_kmeans_cluster_result(all_latent_dicts, config, ensemble_dict)
                elif config['post_fuse_method'] == 'hier':
                    # 层次聚类方法，返回聚类结果（与 kmeans 返回结构一致）
                    return self._get_hier_cluster_result(all_latent_dicts, config, ensemble_dict)
                elif config['post_fuse_method'] == 'sim':
                    # 非聚类：按样本逐一与query做相似度（将每个样本当作一个“簇”以复用注入逻辑）
                    return self._get_similarity_result(all_latent_dicts, config, ensemble_dict)
                else:
                    raise ValueError("Unsupported ensemble method! Supported methods: mean, pca, svd, kmeans")
                   
            # reorganize ensemble_dict into layers
            layers = list(all_latent_dicts[0].keys())
            output_dict = {layer: {} for layer in layers} 
            for module, context_vector in ensemble_dict.items():
                for layer_idx, layer in enumerate(layers):
                    output_dict[layer][module] = context_vector[layer_idx, :].detach().to('cpu')  # (d)

        return output_dict

    def _get_kmeans_cluster_result(self, all_latent_dicts, config, ensemble_dict):
        """
        对所有example的token进行kmeans聚类，返回聚类结果
        返回格式: [{layer 0: {module: context_vector}, layer 1: {...},}, 
                   {layer 0: {module:{"cluster": tensor(k,dim), "index": [[1,2,3], ...]}}, ...}]
        """
        import numpy as np
        import random
        
        # 获取聚类参数
        n_clusters = config.get('kmeans_n_clusters', 3)
        random_state = config.get('kmeans_random_state', 42)
        max_iter = config.get('kmeans_max_iter', 100)
        tol = config.get('kmeans_tol', 1e-4)
        
        from sklearn.cluster import KMeans
        
        def euclidean_kmeans(X, n_clusters, max_iter=100, tol=1e-4, random_state=42):
            """
            欧式距离KMeans聚类算法（回退到原始方法）
            
            Args:
                X: (n_samples, n_features) 输入数据
                n_clusters: 聚类数量
                max_iter: 最大迭代次数
                tol: 收敛阈值
                random_state: 随机种子
                
            Returns:
                cluster_centers: (n_clusters, n_features) 聚类中心
                labels: (n_samples,) 聚类标签
            """
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=max_iter, tol=tol)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            return centers, labels
        
        # 第一个字典：正常的context vector（使用现有的ensemble_dict）
        context_vector_dict = {}
        # 第二个字典：聚类结果
        cluster_dict = {}
        
        layers = list(all_latent_dicts[0].keys())
        
        for layer in layers:
            context_vector_dict[layer] = {}
            cluster_dict[layer] = {}
            
            for module in config['module']:
                # 收集所有example的代表token向量进行聚类
                example_vectors = []
                for example_idx, latent_dict in enumerate(all_latent_dicts):
                    latent_value = latent_dict[layer][module]  # (b, seq_len, d)
                    batch_size, seq_len, dim = latent_value.shape
                    
                    # 根据tok_pos选择特定位置的向量
                    if config['tok_pos'] == 'last':
                        example_vector = latent_value[:, -1, :].squeeze()  # 取最后一个token
                    elif config['tok_pos'] == 'label':
                        example_vector = latent_value[:, -2, :].squeeze()  # 取倒数第二个token（标签词）
                    elif config['tok_pos'] == 'first':
                        example_vector = latent_value[:, 0, :].squeeze()   # 取第一个token
                    elif config['tok_pos'] == 'random':
                        example_vector = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()  # 随机取一个token
                    else:
                        raise ValueError("only support last, label, first or random!")
                    
                    example_vectors.append(example_vector.detach().cpu().numpy())
                
                # 转换为numpy数组
                all_tokens = np.array(example_vectors)  # (num_examples, dim)
                
                # 进行欧式距离kmeans聚类（回退到原始方法）
                cluster_centers, cluster_labels = euclidean_kmeans(
                    all_tokens, n_clusters, max_iter, tol, random_state
                )
                
                # 将所有example的向量堆叠，保持所有shots的向量
                all_example_vectors = torch.stack([torch.tensor(vec) for vec in example_vectors], dim=0)  # (num_examples, dim)
                context_vector_dict[layer][module] = all_example_vectors.detach().to('cpu')  # (num_examples, dim)
                
                # 组织聚类结果
                cluster_result = {
                    "cluster": torch.tensor(cluster_centers, dtype=torch.float32),  # (k, dim)
                    "index": []
                }
                
                # 按聚类组织index：每个聚类包含属于该聚类的example索引
                cluster_indices = [[] for _ in range(n_clusters)]  # 为每个聚类创建空列表
                for example_idx in range(len(all_latent_dicts)):
                    cluster_label = cluster_labels[example_idx]  # 该example的代表token属于的聚类
                    cluster_indices[cluster_label].append(example_idx)  # 将example索引添加到对应聚类
                cluster_result["index"] = cluster_indices
                # 统计簇数（非空簇）
                num_clusters = sum(1 for idxs in cluster_indices if len(idxs) > 0)
                cluster_result["num_clusters"] = int(num_clusters)
                print(f"[KMeans] layer {layer}, module {module}: num_clusters = {num_clusters}")

                cluster_dict[layer][module] = cluster_result

        # print(f"context_vector_dict: {context_vector_dict}")
        # print(f"cluster_dict: {cluster_dict}")
        return [context_vector_dict, cluster_dict]

    def _get_similarity_result(self, all_latent_dicts, config, ensemble_dict):
        """
        非聚类方式：收集每个 example 的代表 token 向量，直接用于与 query 做余弦相似度。
        为了复用已有注入权重计算逻辑，这里将每个 example 视作一个单元素簇：
        cluster = all_example_vectors, index = [[0], [1], ..., [n-1]]。
        返回结构与 kmeans/hier 一致：[context_vector_dict, cluster_dict]
        """
        import numpy as np
        import random

        context_vector_dict = {}
        cluster_dict = {}

        layers = list(all_latent_dicts[0].keys())
        for layer in layers:
            context_vector_dict[layer] = {}
            cluster_dict[layer] = {}

            for module in config['module']:
                example_vectors = []
                for latent_dict in all_latent_dicts:
                    latent_value = latent_dict[layer][module]  # (b, seq_len, d)
                    if config['tok_pos'] == 'last':
                        vec = latent_value[:, -1, :].squeeze()
                    elif config['tok_pos'] == 'label':
                        vec = latent_value[:, -2, :].squeeze()
                    elif config['tok_pos'] == 'first':
                        vec = latent_value[:, 0, :].squeeze()
                    elif config['tok_pos'] == 'random':
                        vec = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                    else:
                        raise ValueError("only support last, label, first or random!")
                    example_vectors.append(vec.detach().cpu())

                all_example_vectors = torch.stack(example_vectors, dim=0)  # (n, d)
                context_vector_dict[layer][module] = all_example_vectors.detach().cpu()

                n = all_example_vectors.size(0)
                cluster_result = {
                    "cluster": all_example_vectors.float(),
                    "index": [[i] for i in range(n)],
                    "num_clusters": int(n),
                }
                print(f"[SIM] layer {layer}, module {module}: num_clusters = {n}")
                cluster_dict[layer][module] = cluster_result

        return [context_vector_dict, cluster_dict]

    def _get_hier_cluster_result(self, all_latent_dicts, config, ensemble_dict):
        """
        使用层次聚类（Agglomerative）对所有 example 的代表 token 做聚类。
        返回结构与 kmeans 分支一致：
        [context_vector_dict, cluster_dict]
        其中 cluster_dict[layer][module] = {"cluster": tensor(k, d), "index": [[...], ...], "num_clusters": k}
        """
        import numpy as np
        import random
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import normalize

        linkage = config.get('hier_linkage', 'average')
        distance_threshold = config.get('hier_threshold', 0.4)
        min_cluster_size = int(config.get('min_cluster_size', 3))

        context_vector_dict = {}
        cluster_dict = {}

        layers = list(all_latent_dicts[0].keys())
        for layer in layers:
            context_vector_dict[layer] = {}
            cluster_dict[layer] = {}

            for module in config['module']:
                # 收集所有 example 的代表 token 向量
                example_vectors = []
                for latent_dict in all_latent_dicts:
                    latent_value = latent_dict[layer][module]
                    if config['tok_pos'] == 'last':
                        vec = latent_value[:, -1, :].squeeze()
                    elif config['tok_pos'] == 'label':
                        vec = latent_value[:, -2, :].squeeze()
                    elif config['tok_pos'] == 'first':
                        vec = latent_value[:, 0, :].squeeze()
                    elif config['tok_pos'] == 'random':
                        vec = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                    else:
                        raise ValueError("only support last, label, first or random!")
                    example_vectors.append(vec.detach().cpu().numpy())

                X = np.array(example_vectors)  # (n, d)
                X_norm = normalize(X, norm='l2')

                # 自适应簇数（通过阈值剪枝）
                model = AgglomerativeClustering(
                    n_clusters=None,
                    metric='cosine',
                    linkage=linkage,
                    distance_threshold=distance_threshold,
                )
                labels = model.fit_predict(X_norm)

                n_clusters_found = int(labels.max() + 1)
                clusters = [[] for _ in range(n_clusters_found)]
                for i, c in enumerate(labels):
                    clusters[c].append(i)

                # 小簇合并（可选）
                if min_cluster_size > 1 and n_clusters_found > 0:
                    valid = [i for i, idxs in enumerate(clusters) if len(idxs) >= min_cluster_size]
                    small = [i for i, idxs in enumerate(clusters) if len(idxs) < min_cluster_size]
                    if len(valid) == 0 and n_clusters_found > 0:
                        valid = list(range(n_clusters_found))
                        small = []
                    if len(small) > 0 and len(valid) > 0:
                        X_t = torch.tensor(X, dtype=torch.float32)
                        X_t = F.normalize(X_t, p=2, dim=-1)
                        cluster_means = []
                        for i in range(n_clusters_found):
                            idxs = clusters[i]
                            if len(idxs) == 0:
                                cluster_means.append(None)
                            else:
                                cluster_means.append(X_t[idxs].mean(dim=0, keepdim=True))
                        for s in small:
                            if len(clusters[s]) == 0:
                                continue
                            sims = []
                            s_mean = cluster_means[s] if cluster_means[s] is not None else X_t[clusters[s]].mean(dim=0, keepdim=True)
                            for v in valid:
                                sims.append(torch.matmul(s_mean, cluster_means[v].t()).item())
                            best = valid[int(np.argmax(np.array(sims)))]
                            clusters[best].extend(clusters[s])
                            clusters[s] = []
                    clusters = [idxs for idxs in clusters if len(idxs) > 0]

                # 计算簇中心（均值）
                centers = []
                for idxs in clusters:
                    if len(idxs) == 0:
                        continue
                    centers.append(torch.tensor(X[idxs]).mean(dim=0, keepdim=False).unsqueeze(0))
                if len(centers) == 0:
                    centers = [torch.tensor(X).mean(dim=0, keepdim=False).unsqueeze(0)]
                    clusters = [list(range(X.shape[0]))]
                centers_t = torch.cat(centers, dim=0).float()

                # 保存结果（与 kmeans 一致）
                all_example_vectors = torch.tensor(X, dtype=torch.float32)
                context_vector_dict[layer][module] = all_example_vectors.detach().cpu()
                cluster_result = {
                    "cluster": centers_t,
                    "index": clusters,
                }
                num_clusters = len([idxs for idxs in clusters if len(idxs) > 0])
                cluster_result["num_clusters"] = int(num_clusters)
                print(f"[HIER] layer {layer}, module {module}: num_clusters = {num_clusters}")
                cluster_dict[layer][module] = cluster_result

        return [context_vector_dict, cluster_dict]


    def _simple_svd_fusion(self, latent_list, topk=None):
        """
        简单的SVD加权融合方法 - 只使用最大奇异值对应的方向
        
        Args:
            latent_list: 形状为 (ensemble_num, layer_num, d) 的张量列表
            
        Returns:
            context_vector: 形状为 (layer_num, d) 的融合后的上下文向量
        """
        # 添加调试信息
        print(f"Debug - latent_list length: {len(latent_list)}")
        print(f"Debug - latent_list[0] shape: {latent_list[0].shape}")
        
        # 将latent_list堆叠成张量
        latents = torch.stack(latent_list, dim=0)  # (ensemble_num, layer_num, d)
        print(f"Debug - latents shape: {latents.shape}")
        ensemble_num, layer_num, d = latents.size()
        print(f"Debug - ensemble_num: {ensemble_num}, layer_num: {layer_num}, d: {d}")
        
        # 为每一层单独计算SVD
        context_vectors = []
        for layer_idx in range(layer_num):
            # 提取当前层的所有示范样本向量
            layer_latents = latents[:, layer_idx, :]  # (ensemble_num, d)
            
            # 计算SVD（稳定版）
            # --- 旧实现（保留备查） ---
            # U, S, V = torch.svd(layer_latents.float())
            # # 使用奇异值作为权重，加权组合右奇异向量
            # weights = S / S.sum()  # 归一化权重 - 使用所有方向
            # context_vector = torch.sum(V * weights.unsqueeze(0), dim=1)  # (d,)
            # # 只使用最大奇异值对应的方向（第一个右奇异向量）
            # # context_vector = V[:, 0]  # 取第一列，得到 (d,) 形状
            # --- 新实现：先在系数空间融合，再映回特征空间 ---
            U, S, Vh = torch.linalg.svd(layer_latents.float(), full_matrices=False)
            V = Vh.t()  # (d, k)
            # 系数矩阵：每个样本在共享基底 V 下的坐标 C = U Σ
            C = U * S.unsqueeze(0)  # (n, k)
            # 系数空间融合策略：均值；若指定 topk，仅使用前K个方向
            if topk is not None:
                k_use = min(int(topk), C.size(1))
                C_use = C[:, :k_use]
                V_use = V[:, :k_use]
                c_merged = C_use.mean(dim=0)
                context_vector = V_use @ c_merged
            else:
                c_merged = C.mean(dim=0)  # (k,)
                context_vector = V @ c_merged  # (d,)
            # 映回特征空间，得到该层的上下文向量
            
            context_vectors.append(context_vector)
        
        # 将所有层的上下文向量堆叠
        final_context_vector = torch.stack(context_vectors, dim=0)  # (layer_num, d)
        print(f"Debug - final_context_vector shape: {final_context_vector.shape}")
        
        return final_context_vector
    
    def _svd_ties_fusion(self, latent_list, tau=0.7, topk=None, agg='mean'):
        """
        重新实现的 TIES 融合：
        - 逐层 SVD：X = U Σ V^T
        - 系数空间 C = U * Σ
        - 符号一致筛选：同号比例 >= tau 的维度保留
        - 聚合：在保留维度上对系数做 mean/median
        - 稀疏化：可选按幅值取前 topk 维
        - 回到特征空间：v = V_sel @ c_merged
        返回 shape (layer_num, d)
        """
        latents = torch.stack(latent_list, dim=0)  # (n, L, d)
        n, L, d = latents.size()
        vectors = []
        for layer_idx in range(L):
            X = latents[:, layer_idx, :].float()  # (n, d)
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            V = Vh.t()  # (d, r)
            C = U * S.unsqueeze(0)  # (n, r)
            # 符号一致筛选
            agree = (torch.sign(C).sum(dim=0).abs() / n) >= float(tau)
            if agree.any():
                C_sel = C[:, agree]
                V_sel = V[:, agree]
            else:
                C_sel = C
                V_sel = V
            # 聚合
            c_merged = C_sel.median(dim=0).values if agg == 'median' else C_sel.mean(dim=0)
            # 稀疏化
            if topk is not None and c_merged.numel() > int(topk):
                k_use = int(topk)
                idx = torch.topk(c_merged.abs(), k_use, largest=True).indices
                mask = torch.zeros_like(c_merged, dtype=torch.bool)
                mask[idx] = True
                c_merged = c_merged * mask
                V_sel = V_sel[:, mask]
            v = V_sel @ c_merged  # (d,)
            vectors.append(v)
        return torch.stack(vectors, dim=0)  # (L, d)
    def _get_cluster_context_vector(self, output, context_vector, cluster_info, cluster_index, using_cluster=False, config=None):
        """
        Args:
            output: (batch_size, seq_len, dim)
            context_vector: (nums_shot, dim)
            cluster_info: (k, dim)
            cluster_index: list of list, len = k, each list contains indices of shots belonging to that cluster
            using_cluster: bool, whether to use cluster centers or shots for weighting
            config: configuration dictionary for parameters
        Returns:
            context_out: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, dim = output.size()
        k = cluster_info.size(0)

        # 使用mean-pooling获得query的全局表示，减少噪声
        query_pool_method = config.get('query_pool_method', 'mean')  # 'mean', 'last', 'first', 'max'
        
        if query_pool_method == 'mean':
            # Mean pooling: 对所有token取平均
            query_repr = output.mean(dim=1)  # (batch_size, dim)
        elif query_pool_method == 'last':
            # Last token pooling: 使用最后一个token
            query_repr = output[:, -1, :]  # (batch_size, dim)
        elif query_pool_method == 'first':
            # First token pooling: 使用第一个token
            query_repr = output[:, 0, :]  # (batch_size, dim)
        elif query_pool_method == 'max':
            # Max pooling: 取每个维度的最大值
            query_repr = output.max(dim=1)[0]  # (batch_size, dim)
        else:
            raise ValueError(f"Unsupported query_pool_method: {query_pool_method}")

        # 计算query表示与聚类中心的余弦相似度
        query_norm = torch.norm(query_repr, p=2, dim=-1, keepdim=True)  # [batch_size, 1]
        cluster_norm = torch.norm(cluster_info, p=2, dim=-1, keepdim=True)  # [k, 1]
        
        # 计算余弦相似度: [batch_size, k]
        cosine_sim = torch.matmul(query_repr, cluster_info.t()) / (query_norm * cluster_norm.t() + 1e-8)
        
        # 将余弦相似度转换为距离（1 - 余弦相似度）
        dist = 1.0 - cosine_sim  # [batch_size, k]

        # 新方案：仅选最近簇，然后使用该簇内样本均值作为 in-context vector
        # 找到最近簇索引（等价于 argmax cosine_sim）
        best_cluster_idx = torch.argmax(cosine_sim, dim=-1)  # [batch_size]

        if using_cluster:
            # 使用最近簇中心
            context_per_query = cluster_info[best_cluster_idx]
        else:
            # 预计算每个簇内样本均值
            k = cluster_info.size(0)
            means = []
            for i in range(k):
                idx = cluster_index[i]
                if len(idx) == 0:
                    # 空簇回退：用簇中心
                    means.append(cluster_info[i].unsqueeze(0))
                else:
                    means.append(context_vector[idx].mean(dim=0, keepdim=True))
            cluster_means = torch.cat(means, dim=0).to(output.device)  # [k, d]
            context_per_query = cluster_means[best_cluster_idx]  # [batch_size, d]

        # 原方案：对所有簇进行加权融合（已注释）
        # weights = 1.0 / (dist + 1e-8)
        # weights = weights / weights.sum(dim=-1, keepdim=True)
        # if using_cluster:
        #     context_per_query = torch.matmul(weights, cluster_info)
        # else:
        #     nums_shot = context_vector.size(0)
        #     shot_weights = torch.zeros(batch_size, nums_shot, device=output.device)
        #     for i, indices in enumerate(cluster_index):
        #         if len(indices) == 0:
        #             continue
        #         per_shot_w = weights[:, i].unsqueeze(1)
        #         shot_weights[:, indices] += per_shot_w
        #     shot_weights = shot_weights / (shot_weights.sum(dim=-1, keepdim=True) + 1e-8)
        #     context_per_query = torch.matmul(shot_weights, context_vector)

        context_out = context_per_query.unsqueeze(1).expand(-1, seq_len, -1)
        return context_out
    def calibrate_strength(self, context_vector_dict, dataset, config, 
                           save_dir=None, run_name=None):
        # prepare label dict          
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")

        # frozen all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # get all_data
        all_data = dataset.all_data
        

        
        # init optimizer
        optim_paramters = [{'params': self.linear_coef}]

        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'], 
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'], 
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get epochs and batch_size from config
        epochs = config['epochs']
        batch_size = config['grad_bs']
        
        # init lr_scheduler
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05*epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
                    if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # train
        print('Calibrating strength params...')
        with self.inject_latent(context_vector_dict, config,
                                self.linear_coef, train_mode=True):
            loss_list = []
            all_data_index = list(range(len(all_data)))
            epoch_iter = len(all_data) // batch_size
            for _ in range(epochs):
                epoch_loss = []
                for i in range(epoch_iter):
                    np.random.shuffle(all_data_index)
                    batch_index = all_data_index[:batch_size]
                    batch_data = [all_data[idx] for idx in batch_index]
                    batch_input, batch_label = [], []
                    for data in batch_data:
                        input_str, ans_list, label = dataset.apply_template(data)

                        # collect single demonstration example
                        if config['cali_example_method'] == 'normal':
                            pass
                        elif config['cali_example_method'] == 'random_label':
                            label = random.choice(list(range(len(ans_list))))
                        else:
                            raise ValueError("only support normal or random_label!")
                        
                        batch_input.append(input_str)
                        batch_label.append(label)

                    input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                    input_ids = input_tok['input_ids'].to(self.device)
                    attn_mask = input_tok['attention_mask'].to(self.device)
                    pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                    # set global vars
                    gv.ATTN_MASK_END = pred_loc
                    gv.ATTN_MASK_START = torch.zeros_like(pred_loc)
                    # forward
                    logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                    # get prediction logits
                    pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                    # get loss
                    gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                    epoch_loss.append(loss.item())
                    # update strength params
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    cur_lr = optimizer.param_groups[0]['lr']
                    print(f'Epoch {_+1}/{epochs}, batch {i//batch_size+1}/{len(all_data)//batch_size+1}, loss: {loss.item()}, lr: {cur_lr}')
                epoch_loss = np.mean(epoch_loss)
                loss_list.append(epoch_loss)

        # fronzen all learnable strength params
        self.linear_coef.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')


    def softprompt(self, config, dataset, save_dir=None, run_name=None):
        pt_config = PromptTuningConfig(**config['pt_config'])
        peft_model = get_peft_model(self.model, pt_config)

        # prepare label dict          
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")

        # print trainable parameters
        peft_model.print_trainable_parameters()
        print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model

        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'], 
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data

        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05*epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
                    if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # train
        loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                for data in batch_data:
                    input_str, _, label = dataset.apply_template(data)
                    batch_input.append(input_str)
                    batch_label.append(label)

                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                # forward
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())
                # update strength params
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)

        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        

    def init_strength(self, config):
        # get linear_coef size
        if type(config['layer']) == str:
            if config['layer'] == 'all':
                layers = list(range(self.num_layers))
                layer_dim = len(layers)
            elif config['layer'] == 'late':
                layers = list(range((self.num_layers*2)//3, self.num_layers))
                layer_dim = len(layers)
            elif config['layer'] == 'early':
                layers = list(range(self.num_layers//3))
                layer_dim = len(layers)
            elif config['layer'] == 'mid':
                layers = list(range(self.num_layers//3, (self.num_layers*2)//3))
                layer_dim = len(layers)
        elif type(config['layer']) == list:
            layers = config['layer']
            layer_dim = len(layers)
        else:
            raise ValueError("layer must be all, late, early, mid or a list of layer index!")

        if config['inject_method'] == 'add':
            param_size = (layer_dim, len(config['module']), 1)  # (layer_num, module_num, 1)
        elif config['inject_method'] in ['linear', 'balance']:
            param_size = (layer_dim, len(config['module']), 2)  # (layer_num, module_num, 2)
        elif config['inject_method'] == 'static_add':
            # static_add不需要参数，但需要设置inject_layers
            self.linear_coef = None
            self.inject_layers = layers
            print("static_add: no parameters needed")
            return
        else:
            raise ValueError("only support add, linear, balance or static_add!")
        # set inject_layers
        self.inject_layers = layers
        # init linear_coef
        linear_coef = torch.zeros(param_size, device=self.device) 
        linear_coef += torch.tensor(config['init_value'], device=self.device)
        self.linear_coef = nn.Parameter(linear_coef)
        print(f"linear_coef shape: {self.linear_coef.shape}\n")
        if not self.linear_coef.is_leaf:
            raise ValueError("linear_coef is not a leaf tensor, which is required for optimization.")
        

    def init_noise_context_vector(self, context_vector_dict):
        # init learnable context_vector
        for layer, sub_dict in context_vector_dict.items():
            for module, latent in sub_dict.items():
                noise_vector = torch.randn_like(latent).detach().cpu()
                context_vector_dict[layer][module] = noise_vector
        return 