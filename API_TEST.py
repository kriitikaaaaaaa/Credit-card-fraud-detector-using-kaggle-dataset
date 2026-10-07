import requests

def Test_add(url):
    print("-" * 50)
    print("Sending the results to the server...")
    response = requests.post(url)

    if response.status_code == 200:
        print("CSV data processed successfully!")
    else:
        print(f"Error processing CSV: {response.text}")


def Test_get(url):
    print("-" * 50)
    print("Getting the results from the server...")
    response = requests.get(url)
    print(response)

    results = response.json()

    print("Results from the server:" + "\n")
    print(results)


def Test_delete(url):
    print("-" * 50)
    print("Deleting the results from the server...")
    response = requests.delete(url)

    if response.status_code == 200:
        print("Results deleted successfully!")
    else:
        print(f"Error deleting results: {response.text}")


def Test_update(url):
    print("-" * 50)
    print("Updating the results in the server...")
    response = requests.put(url)

    if response.status_code == 200:
        print("Results updated successfully!")
    else:
        print(f"Error updating results: {response.text}")


if __name__ == '__main__':
    print("Testing the API...")

    url = "http://127.0.0.1:5000"

    add_url = url + "/add"
    Test_add(add_url)

    get_url = url + "/results"
    Test_get(get_url)

    delete_url = url + "/delete"
    response = requests.delete(delete_url)

    update_url = url + "/update"
    Test_update(update_url)
