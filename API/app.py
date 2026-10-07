from flask import Flask, request, jsonify
from db import Connect_to_MongoDB, Create_Database
from bson.json_util import dumps
from bson.json_util import loads

app = Flask(__name__)

client = Connect_to_MongoDB()
db = client["TestDB"]


def save_ml_results(results):
    collection = db.get_collection("ML")  # Replace with actual collection

    # Insert the results' dictionary into the collection
    try:
        collection.insert_one(results)
        print("Machine Learning results saved to database.")
    except Exception as e:
        print(f"Error saving results to database: {str(e)}")

# Create a new route to receive the results from the client and save them to the database
@app.route('/add', methods=['POST'])
def add():
    # Get the results from the request
    results = request.json

    save_ml_results(results)

    return jsonify({"message": "Results saved to database."})

# Create a new route to retrieve the results from the database
@app.route('/results', methods=['GET'])
def get_results():
    # Get the collection from the database
    collection = db.get_collection("ML")
    print(collection)

    # Find all the documents in the collection
    cursor = collection.find()

    results = loads(dumps(cursor))

    # Return the results as JSON but exclude the _id field
    results = [{"filename": result["filename"], "results": result["results"], "type": result["type"]} for result in results]
    print(results)
    return jsonify(results)

# Create a new route to delete all the results from the database or a specific result
@app.route('/delete', methods=['DELETE'])
def delete():
    # Get the collection from the database
    collection = db.get_collection("ML")

    # Check if the user wants to delete all the results or a specific result
    if request.args.get('filename'):
        # Delete a specific result based on the filename
        filename = request.args.get('filename')
        collection.delete_one({"filename": filename})
        return jsonify({"message": f"Result with filename {filename} deleted."})
    else:
        # Delete all the results
        collection.delete_many({})
        return jsonify({"message": "All results deleted."})

# Create a new route to update a specific result in the database
@app.route('/update', methods=['PUT'])
def update():
    # Get the collection from the database
    collection = db.get_collection("ML")

    # Get the filename and the new results from the request
    filename = request.json['filename']
    new_results = request.json['results']

    print("Updating the file with filename: ", filename)

    # Update the result with the new results
    collection.update_one({"filename": filename}, {"$set": {"results": new_results}})

    return jsonify({"message": f"Result with filename {filename} updated."})

# Create a new route to list all the collections in the database
@app.route('/collections', methods=['GET'])
def get_collections():
    # Get the list of collections in the database
    collections = db.list_collection_names()

    return jsonify(collections)

# Create a new route to list all the databases in the MongoDB server
@app.route('/databases', methods=['GET'])
def get_databases():
    # Get the list of databases in the MongoDB server
    databases = client.list_database_names()

    return jsonify(databases)

# Create a new route to list all the functions available in the API
@app.route('/functions', methods=['GET'])
def get_functions():
    # Get the list of functions available in the API
    functions = {
        "add": "Add a new result to the database\nParameters: filename, results, type",
        "results": "Get all the results from the database\nParameters: None",
        "delete": "Delete all the results from the database(if no filename is specified) or a specific result\nParameters: filename",
        "update": "Update a specific result in the database\nParameters: filename, results",
        "collections": "List all the collections in the database\nParameters: None",
        "databases": "List all the databases in the MongoDB server\nParameters: None",
        "functions": "List all the functions available in the API\nParameters: None"
    }

    return jsonify(functions)


if __name__ == '__main__':

    app.run(debug=True)
