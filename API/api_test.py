import requests
import ML

Data_path = "Data/"
Results_path = "ML_Results/"

# Define the function to test the model with a CSV file
def Model_Test(file, type):
    # Load the CSV file
    data = ML.read_csv(file)
    X_train, X_test, y_train, y_test = ML.Preprocessing(data, train_data=True)
    results = ML.Machine_Learning_Json_output(X_train, X_test, y_train, y_test, file)
    results = {"filename": file, "results": results, "type": type}

    return results

# Define the function to send the results to the server
def Test_add(results, url):
    print("-" * 50)
    print("Sending the results to the server...")
    # Send the results to the server
    response = requests.post(url, json=results)

    if response.status_code == 200:
        print("CSV data processed successfully!")
    else:
        print(f"Error processing CSV: {response.text}")


def Test_get(url):
    print("-" * 50)
    print("Getting the results from the server...")
    # Get the results from the server
    response = requests.get(url)
    print(response)

    # get the results from the response
    results = response.json()

    # Save the results to a file

    # If results are empty, print a message
    if not results:
        print("No results found in the database.")
        return

    with open(Results_path + "results.json", "w", encoding="utf-8") as file:
        for result in results:
            file.write(str(result) + "\n")
            file.write("_" * 50 + "\n")

    # Print the results
    print("Results from the server:" + "\n")
    print(results)


def Test_delete(url):
    print("-" * 50)
    print("Deleting the results from the server...")
    # Delete the results from the server
    response = requests.delete(url)

    if response.status_code == 200:
        print("Results deleted successfully!")
    else:
        print(f"Error deleting results: {response.text}")


def Test_collections(url):
    print("-" * 50)
    print("Getting the collections from the server...")
    # Get all the collections from the server
    response = requests.get(url)
    print(response)

    # Get the collections from the response
    collections = response.json()

    # Print the collections
    print("Collections from the server:" + "\n")
    print(collections)


def Test_update(results, filename, url):
    print("-" * 50)
    print("Updating the results in the server...")
    # Update the results in the server
    response = requests.put(url, json={"filename": filename, "results": results})

    if response.status_code == 200:
        print("Results updated successfully!")
    else:
        print(f"Error updating results: {response.text}")


def Test_databases(url):
    print("-" * 50)
    print("Getting the databases from the server...")
    # Get all the databases from the server
    response = requests.get(url)
    print(response)

    # Get the databases from the response
    databases = response.json()

    # Print the databases
    print("Databases from the server:" + "\n")
    print(databases)


def Test_get_functions(url):
    print("-" * 50)
    print("Getting the functions from the server...")
    # Get all the functions from the server
    response = requests.get(url)
    print(response)

    # Get the functions from the response
    functions = response.json()

    # Print the functions
    for function in functions:
        print(function + ": " + functions[function] + "\n")


if __name__ == '__main__':
    print("Testing the API...")

    url = "http://127.0.0.1:5000"
    file = Data_path + "VerıKumesı_test.xlsx"

    # Test the model with the CSV file
    results = Model_Test(file, type="test")

    # Send the results to the server
    add_url = url + "/add"
    Test_add(results, add_url)

    # Get the results from the server
    get_url = url + "/results"
    Test_get(get_url)

    # Delete the results from the server
    # delete_url = url + "/delete"
    # response = requests.delete(delete_url)

    # Test get all collections
    collections_url = url + "/collections"
    Test_collections(collections_url)

    # Test get all databases
    databases_url = url + "/databases"
    Test_databases(databases_url)

    # Test get functions of the API
    functions_url = url + "/functions"
    Test_get_functions(functions_url)

    # TODO: Doesn't work as intended
    # Test update the results
    # update_url = url + "/update"
    # Test_update(results, file, update_url)
