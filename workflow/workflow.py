from stage import Stage, Status
import json
from utils import MyThread, MyProcess
import time

class Workflow:
    def __init__(self, config_file, boto3_client_ = None) -> None:
        assert isinstance(config_file, str)
        
        self.workflow_name = None
        self.boto3_client = boto3_client_
        
        self.stages = []
        self.sources = []
        self.sinks = []
        
        config = json.load(open(config_file, 'r'))
        self.parse_config(config)
    
    def parse_config(self, config) -> None:
        num = config['num_stages']
        self.workflow_name = config['workflow_name']
        for i in range(num):
            stage = Stage(self.workflow_name, config[str(i)]['stage_name'], i)
            self.stages.append(stage)
            
        for index, stage in enumerate(self.stages):
            stage.input_files = config[str(index)]['input_files']
            stage.read_pattern = config[str(index)]['read_pattern']
            parents = config[str(index)]['parents']
            for p in parents:
                stage.add_parent(self.stages[p])
            children = config[str(index)]['children']
            for c in children:
                stage.add_child(self.stages[c])
                
        # check dependency
        for stage in self.stages:
            for p in stage.parents:
                assert stage in p.children
                
            for c in stage.children:
                assert stage in c.parents
        
        # select sources and sinks
        for stage in self.stages:
            if len(stage.parents) == 0:
                self.sources.append(stage)

            if len(stage.children) == 0:
                self.sinks.append(stage)
                
        for stage in self.sources:
            stage.status = Status.READY
        
        # check Directed Acyclic Graph
        assert self.check_dag()
        
        
    def check_dag(self):
        queue = self.sources.copy()
        in_degrees = [len(s.parents) for s in self.stages]
        
        count = 0
        while len(queue) > 0:
            node = queue.pop(0)
            count += 1
            
            for child in node.children:
                ids = child.stage_id
                in_degrees[ids] -= 1
                
                if in_degrees[ids] == 0:
                    queue.append(child)
                    
        return count >= len(self.stages)
    
    def check_finished(self, threads):
        assert isinstance(threads, list)
        
        for ids, thread in enumerate(threads):
            if thread is not None and not thread.is_alive():
                self.stages[ids].status = Status.FINISHED
        
        for stage in self.stages:
            if stage.status != Status.FINISHED:
                return False
        return True
    
    def update_stage_status(self):
        for stage in self.stages:
            if stage.status != Status.FINISHED:
                continue
            
            for child in stage.children:
                is_ready = True
                if child.status == Status.WAITING:
                    for parent in child.parents:
                        if parent.status != Status.FINISHED:
                            is_ready = False
                            break
                    if not is_ready:
                        child.status = Status.READY
            
    
    def lazy_execute(self):
        # Stage info is only changed in main thread
        threads = [None for i in range(len(self.stages))]
        
        while not self.check_finished(threads):
            stage = None
            for s in self.stages:
                if s.status == Status.READY:
                    stage = s
                    break
            if stage is None:
                self.update_stage_status()
                continue
            stage.status = Status.RUNNING
            thread = MyThread(target=stage.execute, args=None)
            threads[stage.stage_id] = thread
            thread.start()
            
            self.update_stage_status()
            
        for thread in threads:
            assert not thread.is_alive()
            
        res_list = []
        for thread in threads:
            res_list.append(thread.result)
            
        return res_list
    
    def timeline_execute(self):
        raise NotImplementedError
    
    def eager_execute(self):
        raise NotImplementedError
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
    
    def __del__(self):
        pass


if __name__ == '__main__':
    wf = Workflow( './config.json')
    stage = wf.stages[0]
    stage.num_func = 32
    
    stage = wf.stages[1]
    stage.num_func = 16
    
    stage = wf.stages[2]
    stage.num_func = 16
    
    stage = wf.stages[3]
    stage.num_func = 16
    
    stage = wf.stages[4]
    print(wf.workflow_name, stage)
    stage.status = Status.RUNNING
    stage.num_func = 16
    
    t1 = time.time()
    thread = MyThread(target=stage.execute, args=None)
    thread.start()
    thread.join()
    res = thread.result
    # res = stage.execute()
    t2 = time.time()
    print('Number of functions:', stage.num_func)
    print(t2 - t1)
    for result in res:
        rd = json.loads(result[0])
        print(rd)
        if 'statusCode' not in rd:
            print(rd)
        rd = json.loads(rd['body'])
        print(rd['breakdown'])
        
    print('\n\n')
    