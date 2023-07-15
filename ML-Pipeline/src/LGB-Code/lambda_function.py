import time
import train

def handler(event, context):
    if('dummy' in event) and (event['dummy'] == 1):
        print("Dummy call, doing nothing")
        return
    
    if event['func_id'] == '1':
        num_tasks = int(event['num_tasks'])
        total_trees = int(event['total_trees'])
        task_id = int(event['task_id'])
        
        average_tasks = int(total_trees / num_tasks)
        remain_tasks = total_trees % num_tasks
        
        if task_id < remain_tasks:
            num_processes = average_tasks + 1
        else:
            num_processes = average_tasks
            
        train.train_with_multprocess(task_id, num_processes)
        
    else:
        raise Exception('Invalid func_id')
    
    return {
        'statusCode': 200,
    }