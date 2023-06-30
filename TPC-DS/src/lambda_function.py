import dsq95
import json
import utils

def lambda_handler(event, context):
    
    res = dsq95.invoke_q95_func(event)
    
    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }