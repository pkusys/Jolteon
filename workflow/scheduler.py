from abc import ABC, abstractmethod
from workflow import Workflow
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
    
    @abstractmethod
    def set_parallelism(self, total_parallelism):
        raise NotImplementedError()

class Caerus(Scheduler):
    def __init__(self, workflow: Workflow, storage_mode = 's3'):
        super().__init__(workflow)
        self.storage_mode = storage_mode
        self.parallelism = []
    
    # Get the test datasets
    def profile(self):
        file_size = []
        if self.storage_mode == 's3':
            for stage in self.workflow.stages:
                inputs = stage.input_files
                cnt = 0
                for directory in inputs:
                    cnt += utils.get_dir_size(directory)
                file_size.append(cnt)
                
            sum_file_size = sum(file_size)
            self.parallelism = [item / sum_file_size for item in file_size]
        else:
            raise NotImplementedError()
        
    def set_parallelism(self, total_parallelism):
        num_funcs = [int(item * total_parallelism) for item in self.parallelism]
        for stage in self.workflow.stages:
            stage.num_func = num_funcs[stage.stage_id] if num_funcs[stage.stage_id] > 0 else 1
            
class Orion(Caerus):
    def __init__(self, workflow: Workflow, storage_mode = 's3'):
        super().__init__(workflow)
        self.config = []
        self.max_memory = 1024 * 8
        self.memory_grain = 128
    
    # Get the sampling and some analytical params of the workflow
    def profile(self):
        pass
        
    # Just BFS, but stop and return when one node is statisfied with the latency
    def bestfit(self, latency):
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

                if self.get_latency(new_val) <= latency:
                    return new_val

        config_list = [self.max_memory for i in range(len(self.workflow.stages))]
        return config_list

    def get_cost(self, config_list):
        pass

    def get_latency(self, config_list):
        pass
        
        
if __name__ == '__main__':
    wf = Workflow('config.json')
    scheduler = Caerus(wf)
    scheduler.profile()
    scheduler.set_parallelism(64)
    
    for stage in wf.stages:
        print(str(stage.stage_id) + ':' + str(stage.num_func), end=' ')
    print()
    t1 = time.time()
    res = wf.lazy_execute()
    t2 = time.time()
    print('Time:', t2 - t1)
    infos = []
    time_list = []
    times_list = []
    for ids, r in enumerate(res):
        l = []
        for ids_, result in enumerate(r):
            if ids_ == 0:
                time_list.append(result)
                continue
            info = utils.extract_info_from_log(result[1])
            infos.append(info)
            
            rd = json.loads(result[0])
            if 'statusCode' not in rd:
                print(rd)
            rd = json.loads(rd['body'])
            l.append(rd['breakdown'][-1])
        times_list.append(l)
    cost = 0
    for info in infos:
        cost += info['bill']
    print('Cost:', cost, '$')
    for idx, t in enumerate(time_list):
        print('Stage', idx, 'time:', t)
        print(max(times_list[idx]))
    print('Idea DAG Execution Time:', time_list[0] + time_list[1]\
            + time_list[4] + time_list[6] + time_list[7])
    print('\n\n')