"""
Simple example of querying Elasticsearch creating REST requests
"""
import requests
import json
from pprint import pprint


def search(uri, term):
    """Simple Elasticsearch Query"""
    query = json.dumps({
        "_index":""
    })
    response = requests.get(uri, data=query)
    results = json.loads(response.text)
    return results


def format_results(results):
    """Print results nicely:
    doc_id) content
    """
    data = [doc for doc in results['hits']['hits']]
    for doc in data:
        print("%s) %s" % (doc['_id'], doc['_source']['content']))


def create_doc(uri, doc_data={}):
    """Create new document."""
    query = json.dumps(doc_data)
    response = requests.post(uri, data=query)
    print(response)


if __name__ == '__main__':
    uri_search = 'http://192.168.23.160:9200/si-socials-fb-2017-08'
    uri_create = 'http://192.168.23.160:9200/test/articles/'

    results = search(uri_search, "fox")
    pprint(results)
    # format_results(results)

    # create_doc(uri_create, {"content": "The fox!"})
    # results = search(uri_search, "fox")
    # format_results(results)



