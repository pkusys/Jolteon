from abc import ABC, abstractmethod
import boto3
import utils
import time
import json
import math
import numpy as np

from workflow import Workflow
from perf_model_dist import eq_vcpu_alloc
from utils import MyQueue

# scheduler is responsible for tuning the launch time,
# number of function invocation and resource configuration
# for each stage
class Scheduler(ABC):
    def __init__(self, workflow: Workflow):
        self.workflow = workflow

class Caerus(Scheduler):
    def __init__(self, workflow: Workflow, storage_mode = 's3'):
        super().__init__(workflow)
        self.storage_mode = storage_mode
        self.parallelism_ratio = []
        self.num_funcs = None
    
    # Get the test datasets
    def comp_ratio(self):
        file_size = []
        if self.storage_mode == 's3':
            for stage in self.workflow.stages:
                if stage.allow_parallel is False:
                    file_size.append(0)
                    continue
                inputs = stage.input_files
                cnt = 0
                for directory in inputs:
                    cnt += utils.get_dir_size(directory)
                file_size.append(cnt)
                
            sum_file_size = sum(file_size)
            self.parallelism_ratio = [item / sum_file_size for item in file_size]
        else:
            raise NotImplementedError()
        
    def set_config(self, total_parallelism):
        num_funcs = [int(item * total_parallelism) for item in self.parallelism_ratio]
        num_funcs = [item if item > 0 else 1 for item in num_funcs]
        for stage in self.workflow.stages:
            stage.num_func = num_funcs[stage.stage_id]
        self.num_funcs = num_funcs
            
class Orion(Caerus):
    def __init__(self, workflow: Workflow, storage_mode = 's3'):
        super().__init__(workflow)
        self.config = []
        self.max_memory = 1024 * 8
        self.memory_grain = 512
        
    def set_config(self, total_parallelism, latency, confidence):
        num_funcs = [int(item * total_parallelism) for item in self.parallelism_ratio]
        num_funcs = [item if item > 0 else 1 for item in num_funcs]
        for stage in self.workflow.stages:
            stage.num_func = num_funcs[stage.stage_id]
        self.num_funcs = num_funcs
        
        memory_config = self.bestfit(latency, confidence)
        
        self.workflow.update_workflow_config(memory_config, self.num_funcs)
        
    # Just BFS, but stop and return when one node is statisfied with the latency
    def bestfit(self, latency, confidence):
        # 128 is a configurable value
        memory_grain = self.memory_grain
        config_list = [memory_grain for i in range(len(self.workflow.stages))]

        search_spcae = MyQueue()
        search_spcae.push(config_list)

        while len(search_spcae) > 0:
            val = search_spcae.pop()

            for i in range(len(val)):
                new_val = val.copy()
                new_val[i] += memory_grain
                # Max limit
                if new_val[i] > self.max_memory:
                    continue
                search_spcae.push(new_val)
                
                dist = self.get_distribution(new_val)
                print(self.num_funcs, new_val, dist.probility(latency))
                print(dist)
                print('\n\n\n')
                if dist.probility(latency) >= confidence:
                    return new_val

        config_list = [self.max_memory for i in range(len(self.workflow.stages))]
        return config_list

    def get_distribution(self, config_list):
        vcpus = []
        for idx, memory in enumerate(config_list):
            vcpus.append(eq_vcpu_alloc(memory, self.num_funcs[idx]))
            
        for stage in self.workflow.stages:
            stage.perf_model.set_func_size(vcpus[stage.stage_id])
            
        dist = None
        for stage in self.workflow.sinks:
            tmp_dist = stage.perf_model.calculate()
            if dist is None:
                dist = tmp_dist
            else:
                dist.combine(tmp_dist, 1)
                
        return dist

class Ditto(Scheduler):
    def __init__(self, workflow: Workflow, storage_mode = 's3'):
        super().__init__(workflow)
        self.storage_mode = storage_mode
        self.parallelism_ratio = [None for i in range(len(self.workflow.stages))]
        self.num_funcs = None
        
        # Init the virtual DAG
        self.virtual_stages = []
        self.sources = []
        self.sinks = []
        for stage in self.workflow.stages:
            param_a, _ = stage.perf_model.get_params()
            new_stage = self.Virtual_Stage(param_a, stage.stage_id)
            self.virtual_stages.append(new_stage)
            if stage in self.workflow.sources:
                self.sources.append(new_stage)
            if stage in self.workflow.sinks:
                self.sinks.append(new_stage)
                
        # Build the virtual DAG
        for stage in self.workflow.stages:
            idx = stage.stage_id
            for p in stage.parents:
                self.virtual_stages[idx].add_parent(self.virtual_stages[p.stage_id])
                
            for c in stage.children:
                self.virtual_stages[idx].add_child(self.virtual_stages[c.stage_id])
                
        # Tune the virtual DAG, delete the redundant edges
        for stage in self.virtual_stages:
            maintain_list = []
            for idx, c in enumerate(stage.children):
                if stage.max_distance(c) == 1:
                    maintain_list.append(idx)

            new_children = [stage.children[idx] for idx in maintain_list]
            for idx, c in enumerate(stage.children):
                if idx in maintain_list:
                    continue
                c.parents.remove(stage)
            stage.children = new_children
        
    class Virtual_Stage:
        def __init__(self, param_a, idx = None):
            self.param_a = param_a
            self.stage_id = idx
            self.parents = []
            self.children = []
            
            self.merge_type = None     # 0 is parent-child, 1 is two siblings
            self.merge_stages = None
            self.merge_ratio = None
            
        def add_parent(self, parent):
            assert isinstance(parent, Ditto.Virtual_Stage)
            self.parents.append(parent)
            
        def add_child(self, child):
            assert isinstance(child, Ditto.Virtual_Stage)
            self.children.append(child)
            
        def check_parent(self, parent):
            assert isinstance(parent, Ditto.Virtual_Stage)
            if len(self.parents) == 0:
                return False
            if parent in self.parents:
                return True
            for p in self.parents:
                if p.check_parent(parent):
                    return True
            return False
                
        def check_child(self, child):
            assert isinstance(child, Ditto.Virtual_Stage)
            if len(self.child) == 0:
                return False
            if child in self.children:
                return True
            for c in self.children:
                if c.check_child(child):
                    return True
            return False
        
        # Only support 'v_stage' is a child stage of self
        def max_distance(self, v_stage):
            assert isinstance(v_stage, Ditto.Virtual_Stage)
            if v_stage == self:
                return 0
            
            ret = -1
            for c in self.children:
                val = c.max_distance(v_stage)
                if val >= 0:
                    ret = max(ret, val + 1)
            return ret
        
        def reverse_deepest_stages(self):
            if len(self.parents) == 0:
                return [self]
            
            ret = []
            for c in self.parents:
                val = c.reverse_deepest_stages()
                ret.extend(val)
            
            return ret
        
        def __str__(self):
            return 'stage index: ' + str(self.stage_id)
                    
    
    @staticmethod   
    def merge(stage0, stage1, merge_type):
        assert isinstance(stage0, Ditto.Virtual_Stage)
        assert isinstance(stage1, Ditto.Virtual_Stage)
        
        if merge_type == 0:
            a0 = stage0.param_a
            a1 = stage1.param_a
            new_a = math.sqrt(a0) + math.sqrt(a1)
            new_a = new_a * new_a
            
            ratio = a0 / a1
            ratio = math.sqrt(ratio)
            
            new_stage = Ditto.Virtual_Stage(new_a)
            new_stage.merge_type = 0
            new_stage.merge_stages = [stage0, stage1]
            new_stage.merge_ratio = ratio
            
            return new_stage
        elif merge_type == 1:
            a0 = stage0.param_a
            a1 = stage1.param_a
            
            new_a = a0 + a1
            
            ratio = a0 / a1
            
            new_stage = Ditto.Virtual_Stage(new_a)
            new_stage.merge_type = 1
            new_stage.merge_stages = [stage0, stage1]
            new_stage.merge_ratio = ratio
            
            return new_stage
        else:
            raise ValueError('Invalid merge type')
    
    # stage0 is the parent stage, stage1 is the child stage
    @staticmethod
    def distance(stage0, stage1):
        if stage0 == stage1:
            return 0
        
        ret = -1
        for c in stage0.children:
            val = Ditto.distance(c, stage1)
            if val >= 0:
                ret = min(ret, val + 1) if ret >= 0 else val + 1
        
        return ret
    
    # Assign the degree of parallelism to each stage
    # 'v_stage' is the final stage of the virtual DAG   
    def assign(self, v_stage, degree):
        assert isinstance(v_stage, Ditto.Virtual_Stage)
        assert degree > 0
        
        if v_stage.stage_id is not None:
            self.parallelism_ratio[v_stage.stage_id] = degree
            return
        
        con_stages = v_stage.merge_stages
        stage0 = con_stages[0]
        stage1 = con_stages[1]
        
        assert isinstance(stage0, Ditto.Virtual_Stage)
        assert isinstance(stage1, Ditto.Virtual_Stage)
        
        ratio1 = 1 / (v_stage.merge_ratio + 1)
        ratio0 = 1 - ratio1
        
        degree0 = degree * ratio0
        degree1 = degree * ratio1
            
        self.assign(stage0, degree0)
        self.assign(stage1, degree1)
        
        
    # Get the ratio of the entire workflow
    def comp_ratio(self):
        if len(self.sinks) != 1:
            raise ValueError('Do not support the current virtual DAG in Ditto')
        
        sink_stage = self.sinks[0]
        final_stage = None
        
        stop = True
        while stop:
            idx = None
            dis_val = -1
            leaves = sink_stage.reverse_deepest_stages()
            for v_stage in leaves:
                dis = Ditto.distance(v_stage, sink_stage)
                if dis > dis_val:
                    dis_val = dis
                    idx = leaves.index(v_stage)
            assert dis_val >= 0
            cur_stage = leaves[idx]
            
            if cur_stage == sink_stage:
                stop = False
                final_stage = cur_stage
                break
            
            assert len(cur_stage.children) <= 1
            if len(cur_stage.children) == 0:
                final_stage = cur_stage
                break
            
            c_stage = cur_stage.children[0]
            sib_stages = c_stage.parents
            merge_stage = None
            
            # Merge sibling stages
            for idx, s in enumerate(sib_stages):
                if idx == 0:
                    merge_stage = s
                    continue
                
                merge_stage = Ditto.merge(merge_stage, s, 1)
            
            # Merge parent-child stage
            merge_stage = Ditto.merge(merge_stage, c_stage, 0)
            
            if c_stage == sink_stage:
                sink_stage = merge_stage
            
            for c in c_stage.children:
                c.parents.remove(c_stage)
                c.parents.append(merge_stage)
                merge_stage.children.append(c)

        self.assign(final_stage, 1)
        
    def set_config(self, total_parallelism):
        pr = self.parallelism_ratio.copy()
        for idx, stage in enumerate(self.workflow.stages):
            if stage.allow_parallel is False:
                pr[idx] = 0
        pr =(np.array(pr) / np.sum(pr)).tolist()
        num_funcs = [int(item * total_parallelism) for item in pr]
        num_funcs = [item if item > 0 else 1 for item in num_funcs]
        for stage in self.workflow.stages:
            stage.num_func = num_funcs[stage.stage_id]
        self.num_funcs = num_funcs
        
        
if __name__ == '__main__':
    # Test ditto
    wf = Workflow('ML-pipeline.json', perf_model_type = 2)
    wf.train_perf_model('/home/ubuntu/workspace/chaojin-dev/serverless-bound/profiles/ML-Pipeline_profile.json')
    scheduler = Ditto(wf)
    scheduler.comp_ratio()
    scheduler.set_config(32)
    print(scheduler.parallelism_ratio)
    print(scheduler.num_funcs)
    
    # Test orion
    # wf = Workflow('ML-pipeline.json', perf_model_type = 1)
    # wf.train_perf_model('/home/ubuntu/workspace/chaojin-dev/serverless-bound/profiles/ML-Pipeline_profile.json')
    # scheduler = Orion(wf)
    # scheduler.comp_ratio()
    # scheduler.set_config(32, 25, 0.9)
    # wf.close_pools()
    
    # Test caerus
    # for stage in wf.stages:
    #     print(str(stage.stage_id) + ':' + str(stage.num_func), end=' ')
    # print()
    # t1 = time.time()
    # res = wf.lazy_execute()
    # t2 = time.time()
    # print('Time:', t2 - t1)
    # infos = []
    # time_list = []
    # times_list = []
    # for ids, r in enumerate(res):
    #     l = []
    #     for ids_, result in enumerate(r):
    #         if ids_ == 0:
    #             time_list.append(result)
    #             continue
    #         info = utils.extract_info_from_log(result[1])
    #         infos.append(info)
            
    #         rd = json.loads(result[0])
    #         if 'statusCode' not in rd:
    #             print(rd)
    #         rd = json.loads(rd['body'])
    #         l.append(rd['breakdown'][-1])
    #     times_list.append(l)
    # cost = 0
    # for info in infos:
    #     cost += info['bill']
    # print('Cost:', cost, '$')
    # for idx, t in enumerate(time_list):
    #     print('Stage', idx, 'time:', t)
    #     print(max(times_list[idx]))
    # print('Idea DAG Execution Time:', time_list[0] + time_list[1]\
    #         + time_list[4] + time_list[6] + time_list[7])
    # print('\n\n')