import boto3

def get_files(bucket_name, key):
    assert isinstance(key, str)
    
    s3_client = boto3.client('s3')
    
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=key
    )
    
    res = []
    if 'Contents' in response:
        for file in response['Contents']:
            res.append(file['Key'])
    else:
        raise Exception('No files found')
    return res

def get_suffix(filename):
    assert isinstance(filename, str)
    fn = filename.split('.')[0]
    fn = fn.split('_')
    file_id = fn[-2]
    chunk_id = fn[-1]
    
    return file_id, chunk_id

def get_suffix_str(filename):
    assert isinstance(filename, str)
    fn = filename.split('.')[0]
    fn = fn.split('_')
    file_id = fn[-2]
    chunk_id = fn[-1]
    
    return file_id + '_' + chunk_id