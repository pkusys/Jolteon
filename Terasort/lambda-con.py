import boto3
import json
import time
import base64
import io
from multiprocessing import Pool

# st = time.time()

# MapReduce configuration
num_partitions = 256
num_mappers = 16
num_reducers = 8
func_type = 0  # 0: map, 1: reduce

payload = {
    "s3bucket_in": "serverless-bound", "s3key_in": "terasort/test/10g/test-10g",
    "s3bucket_out": "serverless-bound", "s3key_out":"terasort/test/10g/test-10g",
    "num_mappers": num_mappers, "num_reducers": num_reducers, "num_partitions": num_partitions, 
    "func_type": "map", "task_id": 0
}

def invoke_lambda(idx: int):
    st = time.time()
    client = boto3.client('lambda')

    payload['task_id'] = idx

    response = client.invoke(
        FunctionName='terasort-1',
        LogType='Tail',
        Payload=json.dumps(payload),
        # Qualifier='0',
    )
    et = time.time()

    log_result = response['ResponseMetadata']['HTTPHeaders']['x-amz-log-result']
    log_result = base64.b64decode(log_result)
    log_result = log_result.decode('utf8')

    log_data = log_result.strip().replace('\n', ' ').replace('\t', ' ').split(' ')

    res = {}
    duration_cnt = 0
    for i in range(len(log_data)):
        if log_data[i] == 'Duration:' and duration_cnt == 0:
            res['duration'] = float(log_data[i+1])
            duration_cnt += 1
        elif log_data[i] == 'Billed':
            res['billed_duration'] = float(log_data[i+2])
        elif log_data[i] == 'Size:':  # memory size
            res['memory_size'] = float(log_data[i+1])
        elif log_data[i] == 'Max':
            res['max_memory_used'] = float(log_data[i+3])

    # transfer response.payload to string
    resp_payload = response['Payload'].read()
    resp_payload = resp_payload.decode('utf8')
    resp_payload = resp_payload.replace('\n', ' ').replace('\t', ' ').split(' ')
    for i in range(len(resp_payload)):
        if resp_payload[i] == 'parse_duration':
            res['parse_duration'] = float(resp_payload[i+1])
        elif resp_payload[i] == 'read_duration':
            res['read_duration'] = float(resp_payload[i+1])
        elif resp_payload[i] == 'record_creation_duration':
            res['record_creation_duration'] = float(resp_payload[i+1])
        elif resp_payload[i] == 'sort_duration':
            res['sort_duration'] = float(resp_payload[i+1])
        elif resp_payload[i] == 'write_duration':
            res['write_duration'] = float(resp_payload[i+1])
        elif resp_payload[i] == 'num_mappers':
            res['num_mappers'] = int(resp_payload[i+1])
        elif resp_payload[i] == 'num_reducers':
            res['num_reducers'] = int(resp_payload[i+1])

    # Write log to file
    with open('./logs/terasort-{}-task{}.log'.format(payload["func_type"], idx), 'w') as f:
        f.write(str(et - st) + " s\n\n")
        f.write(log_result)
        f.write('\n\n')
        f.write(str(res))
    
    return res

def test(i: int):
    # time.sleep(0)
    pass

if __name__ == '__main__':
    
    num = num_mappers if func_type == 0 else num_reducers
    payload['func_type'] = "map" if func_type == 0 else "reduce"

    pool = Pool(num)
    # initialize
    li = [i for i in range(num)]
    res = pool.map_async(test, li)
    res.wait()

    st = time.time()
    li = [i for i in range(num)]
    res = pool.map_async(invoke_lambda, li)
    res.wait()
    et = time.time()
    with open('./logs/terasort.log', 'w') as f:
        f.write("end-to-end latency: " + str(et - st) + " s\n\n")

    pool.close()
    pool.join()