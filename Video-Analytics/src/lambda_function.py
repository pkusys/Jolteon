import time
import json

import split
from utils import get_files

bucketName = 'serverless-bound'

def handler(event, context):
    if('dummy' in event) and (event['dummy'] == 1):
        print("Dummy call, doing nothing")
        return
    
    if event['func_id'] == 0:
        num_tasks = int(event['num_tasks'])        
        task_id = int(event['task_id'])
        
        input_address_str = event['input_address']
        output_address = event['output_address']
        
        input_address = get_files(bucketName, input_address_str)
        
        number = len(input_address)
        
        average_tasks = int(number / num_tasks)
        remain_tasks = number % num_tasks
        
        if task_id < remain_tasks:
            num_files = average_tasks + 1
        else:
            num_files = average_tasks
        
        start_file_id = 0
        for idx in range(task_id):
            if idx < remain_tasks:
                start_file_id += average_tasks + 1
            else:
                start_file_id += average_tasks
        
        split_inputs = input_address[start_file_id:start_file_id + num_files]
        
        res = split.split_videos(split_inputs, output_address, task_id, start_file_id)
        
    elif event['func_id'] == 1:
        raise NotImplementedError()
    elif event['func_id'] == 2:
        raise NotImplementedError()
    elif event['func_id'] == 3:
        raise NotImplementedError()
    else:
        raise Exception('Invalid func_id')
    
    return_val = {
        'breakdown': res
    }
    
    return {
        'statusCode': 200,
        'body': json.dumps(return_val)
    }
    
    
if __name__ == '__main__':
    event = {
        'func_id': 0,
        'num_tasks': 1,
        'num_vcpu': 4,
        'task_id': 0,
        'input_address': 'Video-Analytics/dataset/video',
        'output_address': 'Video-Analytics/stage0/clip_video'
    }
    handler(event, None)