import boto3
import json
import decimal
import numpy as np

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

table = dynamodb.Table('Final_State')

test = list(np.random.rand(10))
key = np.random.randint(0,1e10)

table.put_item(
               Item={'key':key,
               'test':json.dumps(test)
               })
