import boto3
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import json
import base64

def extract_name(name):
    assert isinstance(name, str)
    result = name.rpartition('/')
    extracted_string = result[2]
    return extracted_string

class Stage:
    def __init__(self, workflow_name_, stage_name_, stage_id_, func_name_ = None, boto3_client_ = None) -> None:
        assert isinstance(stage_name_, str)
        assert isinstance(stage_id_, int)
        
        self.stage_name = stage_name_
        self.stage_id = stage_id_
        self.boto3_client = boto3_client_
        self.func_name = func_name_
        self.workflow_name = workflow_name_
        
        self.num_func = 1
        self.config = {'memory': 1024, 'timeout': 360}
        
        self.children = []
        self.parents = []
        self.input_files = []
        self.read_pattern = []
        
    def add_child(self, child):
        self.children.append(child)
        
    def add_parent(self, parent):
        self.parents.append(parent)
        
    def set_config(self, config_):
        assert isinstance(config_, dict)
        
        self.config = config_
        
    def update_lamda_config(self):
        if self.func_name is None:
            wn = self.workflow_name.replace('/', '-')
            self.func_name = wn + '-' + self.stage_name
            
        if self.boto3_client is None:
            self.boto3_client = boto3.client('lambda')
            
        response = self.boto3_client.update_function_configuration(
            FunctionName=self.func_name,
            MemorySize=self.config['memory'],
            Timeout=self.config['timeout']
        )
        
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            return True
        else:
            return False
        
    def register_lambda(self, code_bucket, code_key):
        if self.func_name is None:
            wn = self.workflow_name.replace('/', '-')
            self.func_name = wn + '-' + self.stage_name
            
        if self.boto3_client is None:
            self.boto3_client = boto3.client('lambda')
            
        raise Exception("Please register lambda function through AWS CLI with preinstalled dependencies.")
    
    def invoke_lambda(self, payload):
        if self.func_name is None:
            wn = self.workflow_name.replace('/', '-')
            self.func_name = wn + '-' + self.stage_name
            
        if self.boto3_client is None:
            self.boto3_client = boto3.client('lambda')
            
        response = self.boto3_client.invoke(
            # FunctionName='tpcds-96-stage1',
            FunctionName=self.func_name,
            LogType='Tail',
            Payload=json.dumps(payload),
        )
        
        log_result = response['ResponseMetadata']['HTTPHeaders']['x-amz-log-result']
        log_result = base64.b64decode(log_result)
        log_result = log_result.decode('utf8')
        
        resp_payload = response['Payload'].read()
        resp_payload = resp_payload.decode('utf8')
        
        return [resp_payload, log_result]
        
    def execute(self):
        assert self.num_func > 0
        
        if self.boto3_client is None:
            self.boto3_client = boto3.client('lambda')
            
        num_vcpu = cpu_count()
        
        prefix = ''
        input_address = []
        table_name = []
        read_pattern = []
        output_address = prefix + self.workflow_name + '/' + self.stage_name + '/intermediate'
        storage_mode = 's3'
        num_partitions = [None for i in range(len(self.read_pattern))]
        num_tasks = self.num_func
        func_id = self.stage_id
        
        assert len(self.read_pattern) == len(self.input_files)
        index = 0
        
        for i in range(len(self.read_pattern)):
            input_address.append(prefix + self.input_files[i])
            read_pattern.append(self.read_pattern[i])
            table_name.append(extract_name(self.input_files[i]))
            # read from raw table
            if self.read_pattern[i] == 'read_partial_table' or self.read_pattern[i] == 'read_table':
                num_partitions[i] = 1
            # read from intermediate data
            else:
                num_partitions[i] = self.parents[index].num_func
                index += 1
        
        payload = {
            'task_id': 0,
            'input_address': input_address,
            'table_name': table_name,
            'read_pattern': read_pattern,
            'output_address': output_address,
            'storage_mode': storage_mode,
            'num_tasks': num_tasks,
            'num_partitions': num_partitions,
            'func_id': func_id
        }
            
        # construct payload for each lambda function invocation
        payload_list = []
        
        # A list of return values (log_data and response_data) from lambda functions
        ret_list = []
        
        for i in range(self.num_func):
            payload_cp = payload.copy()
            payload_cp['task_id'] = i
            payload_list.append(payload_cp)
        
        t_pool = ThreadPool(num_vcpu)
        
        ret_list = t_pool.map(self.invoke_lambda, payload_list)
        
        t_pool.close()
        t_pool.join()
        
        return ret_list
    
    def __str__(self):
        return self.stage_name