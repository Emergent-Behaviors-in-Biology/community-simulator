from __future__ import print_function # Python 2/3 compatibility
import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')


table = dynamodb.create_table(
    TableName='Metadata',
    KeySchema=[
        {
            'AttributeName': 'sample-id',
            'KeyType': 'HASH'  #Partition key
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'sample-id',
            'AttributeType': 'N'
        },
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 1,
        'WriteCapacityUnits': 1
    }
)

print("Table status:", table.table_status)

table = dynamodb.create_table(
                              TableName='Final_State',
                              KeySchema=[
                                         {
                                         'AttributeName': 'sample-id',
                                         'KeyType': 'HASH'  #Partition key
                                         }
                                         ],
                              AttributeDefinitions=[
                                                    {
                                                    'AttributeName': 'sample-id',
                                                    'AttributeType': 'N'
                                                    },
                                                    ],
                              ProvisionedThroughput={
                              'ReadCapacityUnits': 1,
                              'WriteCapacityUnits': 1
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
                              'ReadCapacityUnits': 1,
                              'WriteCapacityUnits': 1
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
                              'ReadCapacityUnits': 1,
                              'WriteCapacityUnits': 1
                              }
                              )

print("Table status:", table.table_status)
