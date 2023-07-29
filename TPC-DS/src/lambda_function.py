import dsq95
import json
import utils

def lambda_handler(event, context):
    
    # return {
    #     'statusCode': 200,
    #     'body': json.dumps(event)
    # }
    
    if('dummy' in event) and (event['dummy'] == 1):
        print("Dummy call, doing nothing")
        return
    
    res = dsq95.invoke_q95_func(event)
    
    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }