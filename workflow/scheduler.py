from abc import ABC, abstractmethod
from workflow import Workflow
import boto3
import utils
import time
import json

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
        self.storage_mode = storage_mode
        self.parallelism = []
    
    # Get the sampling and some analytical params of the workflow
    def profile(self):
        pass
        
    def bestfit(self):
        pass
        
    def get_cost(self):
        pass
    
    def get_latency(self):
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