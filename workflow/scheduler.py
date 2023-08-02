from abc import ABC, abstractmethod
from workflow import Workflow
from perf_model_dist import eq_vcpu_alloc
import boto3
import utils
import time
import json
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
    def profile(self):
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
        
        print(memory_config)
        
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
        
if __name__ == '__main__':
    wf = Workflow('ML-pipeline.json', perf_model_type = 1)
    wf.train_perf_model('/home/ubuntu/workspace/chaojin-dev/serverless-bound/profiles/ML-Pipeline_profile.json')
    scheduler = Orion(wf)
    scheduler.profile()
    scheduler.set_config(64, 30, 0.9)
    
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