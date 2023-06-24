from io import BytesIO
import os
import pandas as pd
import boto3

def read_local_table(key):
    location = key['input_address']

    part_data = pd.read_table(location, 
                              delimiter="|", 
                              header=None, 
                              na_values='-')

    return part_data

def count_bytes_io(byte_io, grain=100):
    assert isinstance(byte_io, BytesIO)
    assert isinstance(grain, int)
    
    count = 0
    line_count = 0
    counted_lines = 0
    lines = byte_io.getvalue().decode('gbk').splitlines()
    ret = []

    for line in lines:
        count += len(line.encode()) + len(os.linesep.encode())
        line_count += 1

        if line_count % grain == 0:
            ret.append(count)
            counted_lines = line_count
    
    if counted_lines != line_count:
        ret.append(count)
         
    df = pd.DataFrame({'num_bytes': pd.Series(ret, dtype='int64')})
    return df

def generate_meta_data_local(dir_path):
    dat_files = []
    files = os.listdir(dir_path)
    for file_ in files:
        if file_.endswith('.dat') and os.path.isfile(os.path.join(dir_path, file_)):
            dat_files.append(os.path.join(dir_path, file_))
            
    for fn in dat_files:
        suffix = '_meta'
        meta_loc = fn[:fn.rfind('.')] + suffix + fn[fn.rfind('.'):]
        with open(fn, 'rb') as f:
            byte_data = f.read()
            byte_io = BytesIO(byte_data)
        byte_io.seek(0)
        df = count_bytes_io(byte_io)
        df.to_csv(meta_loc, sep='|', header=False, index=False)
        
        print('Generated meta data for {}'.format(fn))
        
def list_s3_files(bucket_name, dir, client=None):
    if client is None:
        client = boto3.client('s3')
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=dir)
    
    file_names = []
    if 'Contents' in response:
        for obj in response['Contents']:
            if not obj['Key'].endswith('/'):
                file_names.append(obj['Key'])
    
    return file_names
        
def generate_meta_data_s3(bucket, dir):
    client = boto3.client('s3')
    file_names = list_s3_files(bucket, dir, client)
    
    for fn in file_names:
        suffix = '_meta'
        meta_loc = fn[:fn.rfind('.')] + suffix + fn[fn.rfind('.'):]
        response = client.get_object(Bucket=bucket, Key=fn)
        byte_data = response['Body'].read()
        byte_io = BytesIO(byte_data)
        df = count_bytes_io(byte_io)
        del byte_data
        del byte_io
        
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, sep='|', header=False, index=False)
        client.put_object(Bucket=bucket,
                        Key=meta_loc,
                        Body=csv_buffer.getvalue())

if __name__ == '__main__':
    bn = 'serverless-bound'
    di = 'tpcds/test'
    
    generate_meta_data_local('/home/ubuntu/workspace/serverless-bound/TPC-DS/data')
    # generate_meta_data_s3(bn, di)
    # with open('/home/ubuntu/workspace/serverless-bound/TPC-DS/data/ship_mode.dat', 'rb') as f:
    #     byte_data = f.read()
    #     byte_io = BytesIO(byte_data)
    #     lines = byte_io.getvalue().decode('gbk').splitlines()
    #     print(len(lines))