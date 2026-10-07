from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def Connect_to_MongoDB():
    uri = "mongodb+srv://raoofagh:T6bp26wlEf1C7pDp@cluster0.sadjhyu.mongodb.net/?retryWrites=true&w=majority&appName" \
          "=Cluster0 "

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    return client


def Create_Database(client):
    # Get the database name
    db_name = "TestDB"

    # Get the collection name within the database

    # Access the database using the client
    db = client[db_name]

    # Insert a document into the collection
    document = {"name": "John Doe", "age": 30}
    db.init.insert_one(document)

    # Find the document in the collection
    document = db.init.find_one({"name": "John Doe"})
    print(document)

    # Delete the added document to clean up
    db.init.delete_one({"name": "John Doe"})

    print("Empty Database Created!\n")
    print("---------------------------------------------------------------------------------")
    return db
