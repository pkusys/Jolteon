import time
import json

import split
import extract
import preprocess
from utils import get_files, get_suffix

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
        
        res = extract.extrace_frames(split_inputs, output_address)
    elif event['func_id'] == 2:
        num_tasks = int(event['num_tasks'])        
        task_id = int(event['task_id'])
        mod_number = int(event['mod_number'])
        
        input_address_str = event['input_address']
        output_address = event['output_address']
        
        input_address = get_files(bucketName, input_address_str)
        
        adresses = []
        for idx, file in enumerate(input_address):
            file_id, chunk_id = get_suffix(file)
            if int(file_id) % mod_number == task_id:
                adresses.append(file)
        input_address = adresses
        
        
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
        res = preprocess.sharpening_filter(split_inputs, output_address)
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
        "func_id": 2,
        "num_tasks": 1,
        "mod_number": 1,
        "num_vcpu": 4,
        "task_id": 0,
        "input_address": "Video-Analytics/stage1/clip_frame",
        "output_address": "Video-Analytics/stage2/filter_frame"
    }
    handler(event, None)