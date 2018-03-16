from __future__ import print_function # Python 2/3 compatibility
import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

for table_name in ['Final_State','Initial_State','Metadata','Parameters']:
  table = dynamodb.Table(table_name)
  table.delete()