from __future__ import print_function # Python 2/3 compatibility
import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')


table = dynamodb.create_table(
    TableName='Metadata',
    KeySchema=[
        {
            'AttributeName': 'data_ID',
            'KeyType': 'HASH'  #Partition key
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'data_ID',
            'AttributeType': 'N'
        },
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

print("Table status:", table.table_status)

table = dynamodb.create_table(
                              TableName='Final_State',
                              KeySchema=[
                                         {
                                         'AttributeName': 'data_ID',
                                         'KeyType': 'HASH'  #Partition key
                                         }
                                         ],
                              AttributeDefinitions=[
                                                    {
                                                    'AttributeName': 'data_ID',
                                                    'AttributeType': 'N'
                                                    },
                                                    ],
                              ProvisionedThroughput={
                              'ReadCapacityUnits': 5,
                              'WriteCapacityUnits': 5
                              }
                              )

print("Table status:", table.table_status)

table = dynamodb.create_table(
                              TableName='Initial_State',
                              KeySchema=[
                                         {
                                         'AttributeName': 'key',
                                         'KeyType': 'HASH'  #Partition key
                                         }
                                         ],
                              AttributeDefinitions=[
                                                    {
                                                    'AttributeName': 'key',
                                                    'AttributeType': 'N'
                                                    },
                                                    ],
                              ProvisionedThroughput={
                              'ReadCapacityUnits': 5,
                              'WriteCapacityUnits': 5
                              }
                              )

print("Table status:", table.table_status)

table = dynamodb.create_table(
                              TableName='Parameters',
                              KeySchema=[
                                         {
                                         'AttributeName': 'key',
                                         'KeyType': 'HASH'  #Partition key
                                         }
                                         ],
                              AttributeDefinitions=[
                                                    {
                                                    'AttributeName': 'key',
                                                    'AttributeType': 'N'
                                                    },
                                                    ],
                              ProvisionedThroughput={
                              'ReadCapacityUnits': 5,
                              'WriteCapacityUnits': 5
                              }
                              )

print("Table status:", table.table_status)
