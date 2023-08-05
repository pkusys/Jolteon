from perf_model import StagePerfModel
from perf_model_dist import DistPerfModel
import boto3
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from multiprocessing import Pool
import json
import base64
from enum import Enum
import time

# extrace stage from /tpcds/stage/intermediate
def extract_name(name):
    assert isinstance(name, str)
    result = name.rpartition('/')
    extracted_string = result[2]
    if extracted_string == 'intermediate':
        extracted_string = result[0]
        result = extracted_string.rpartition('/')
        extracted_string = result[2]
    return extracted_string

class Status(Enum):
    WAITING = 0
    READY = 1
    RUNNING = 2
    FINISHED = 3
    
class PerfModel(Enum):
    Jolteon = 0
    Distribution = 1
    Analytical = 2

class Stage:
    def __init__(self, workflow_name_, stage_name_, stage_id_, perf_model_type = 0, func_name_ = None) -> None:
        assert isinstance(stage_name_, str)
        assert isinstance(stage_id_, int)
        
        # 0 for Jolteon, 1 for Orion distribution model, 2 for Ditto Caerus Locus analytical model
        assert isinstance(perf_model_type, int)
        
        self.stage_name = stage_name_
        self.stage_id = stage_id_
        self.func_name = func_name_
        self.workflow_name = workflow_name_
        
        self.status = Status.WAITING
        
        self.num_func = 1
        
        self.config = {'memory': 2048, 'timeout': 360}

        if perf_model_type == PerfModel.Jolteon.value:
            self.perf_model = StagePerfModel(self.stage_id, self.stage_name)
        elif perf_model_type == PerfModel.Distribution.value:
            self.perf_model = DistPerfModel(self.stage_id, self.stage_name)
        elif perf_model_type == PerfModel.Analytical.value:
            self.perf_model = None
        else:
            raise ValueError('Invalid performance model type: %d' % perf_model_type)
        
        self.children = []
        self.parents = []
        self.input_files = None
        self.output_files = None
        self.read_pattern = None
        self.extra_args = None
        
        self.allow_parallel = True
        
        # 64 is a magic number, according to you central server's CPU cores
        self.pool_size = 64
        self.pool = Pool(self.pool_size)
        # self.boto3_client = boto3.client('lambda')
    
    def change_pool_size(self, new_size):
        assert isinstance(new_size, int)
        self.pool.close()
        self.pool.join()
        
        self.pool_size = new_size
        self.pool = Pool(self.pool_size)
        
    def add_child(self, child):
        self.children.append(child)
        
    def add_parent(self, parent):
        self.parents.append(parent)
        
    def set_config(self, config_):
        assert isinstance(config_, dict)
        
        self.config = config_
        
    def update_lambda_config(self):
        if self.func_name is None:
            wn = self.workflow_name.replace('/', '-')
            self.func_name = wn + '-' + self.stage_name
            
        boto3_client = boto3.client('lambda')
            
        response = boto3_client.update_function_configuration(
            FunctionName=self.func_name,
            MemorySize=self.config['memory'],
            Timeout=self.config['timeout']
        )
        
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            return True
        else:
            return False

    def update_config(self, new_memory, new_num_func):
        assert isinstance(new_memory, int) and new_memory >= 128 and new_memory <= 10*1024 and \
            isinstance(new_num_func, int)
        if not self.allow_parallel:
            new_num_func = 1

        self.config['memory'] = new_memory
        self.num_func = new_num_func
        
        ret = True
        start_time = time.time()
        # update lambda function configuration util success
        while not self.update_lambda_config():
            if time.time() - start_time > 10:
                ret = False
                break
            
        return ret
        # if self.update_lambda_config():
        #     return True
        # else:
        #     return False
        
    def register_lambda(self, code_bucket, code_key):
        if self.func_name is None:
            wn = self.workflow_name.replace('/', '-')
            self.func_name = wn + '-' + self.stage_name
            
        raise Exception("Please register lambda function through AWS CLI with preinstalled dependencies.")
    
    def invoke_lambda(self, payload):
        if self.func_name is None:
            wn = self.workflow_name.replace('/', '-')
            self.func_name = wn + '-' + self.stage_name

        boto3_client = boto3.client('lambda')
        
        t0 = time.time()
        response = boto3_client.invoke(
            # FunctionName='tpcds-96-stage1',
            FunctionName=self.func_name,
            LogType='Tail',
            Payload=json.dumps(payload),
        )
        t1 = time.time()
        
        log_result = response['ResponseMetadata']['HTTPHeaders']['x-amz-log-result']
        log_result = base64.b64decode(log_result)
        log_result = log_result.decode('utf8')
        
        resp_payload = response['Payload'].read()
        resp_payload = resp_payload.decode('utf8')

        # print('Lambda invocation time: ', t1 - t0)
        
        return [resp_payload, log_result]
        
    def execute(self, dummy=0):
        assert dummy == 0 or dummy == 1
        assert self.status == Status.RUNNING
        if not self.allow_parallel:
            assert self.num_func == 1
        else:
            assert self.num_func > 0
        
        # self.status = Status.RUNNING
        
        prefix = ''
        input_address = []
        output_address = []
        table_name = []
        read_pattern = []
        storage_mode = 's3'
        num_partitions = [None for i in range(len(self.read_pattern))]
        num_tasks = self.num_func
        func_id = self.stage_id
        
        assert len(self.read_pattern) == len(self.input_files)
        index = 0
        
        assert self.input_files is not None
        for i in range(len(self.input_files)):
            if self.input_files is not None:
                input_address.append(prefix + self.input_files[i])
                table_name.append(extract_name(self.input_files[i]))
            if self.read_pattern is not None:
                read_pattern.append(self.read_pattern[i])
            # read from raw table
            if self.read_pattern[i] == 'read_partial_table' or self.read_pattern[i] == 'read_table':
                num_partitions[i] = 1
            # read from intermediate data
            else:
                num_partitions[i] = self.parents[index].num_func
                index += 1
                
        if self.output_files is not None:
            for i in range(len(self.output_files)):
                output_address.append(prefix + self.output_files[i])
        elif self.output_files is None:
            output_address = prefix + self.workflow_name + '/' + self.stage_name + '/intermediate'
        
        # 1792 is ad-hoc value for AWS lambda
        num_vcpu = int(round(self.config['memory'] / 1792))
        num_vcpu = max(num_vcpu, 1)
        
        payload = {
            'task_id': 0,
            'dummy': dummy,
            'input_address': input_address,
            'table_name': table_name,
            'read_pattern': read_pattern,
            'output_address': output_address,
            'storage_mode': storage_mode,
            'num_tasks': num_tasks,
            'num_partitions': num_partitions,
            'func_id': func_id,
            'num_vcpu': num_vcpu
        }
        
        if self.extra_args is not None:
            for k in self.extra_args.keys():
                assert k not in payload.keys()
                payload[k] = self.extra_args[k]
            
        # construct payload for each lambda function invocation
        payload_list = []
        
        # A list of return values (log_data and response_data) from lambda functions
        ret_list = []
        
        for i in range(self.num_func):
            payload_cp = payload.copy()
            payload_cp['task_id'] = i
            payload_list.append(payload_cp)
            
        t0 = time.time()
        
        t_pool = self.pool
        # res = self.invoke_lambda(payload_list[0])
        # ret_list.append(res)
        ret_list = t_pool.map(self.invoke_lambda, payload_list)
        
        t1 = time.time()
        
        # print(self.stage_id, 'Funtion invocation time: ', t1 - t0, 's')
        
        ret_list.insert(0, t1 - t0)
        
        # move it to workflow execution for thread safety
        # self.status = Status.FINISHED
        
        return ret_list
    
    def close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
    
    def __str__(self):
        return self.stage_name
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    '''
        In Python, the __del__ method is called when an object's reference count drops to zero. 
    However, this behavior can lead to problems when your object relies on other objects (for 
    example, your Stage object relies on the multiprocessing.Pool object). If Python's garbage 
    collector deletes the pool object first and then deletes the Stage object, then calling 
    self.pool.close() in the __del__ method will raise an AttributeError.
    '''
    # def __del__(self):
    #     self.pool.close()
    #     self.pool.join()