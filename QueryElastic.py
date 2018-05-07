from elasticsearch import Elasticsearch
import json

MAX_SCROLL = 50000000
hosts = [{"host": "192.168.23.160", "port": 9200}]
es = Elasticsearch(hosts= hosts)
batch_size = 10000
res = es.search(index="si-socials-fb-2018-03",size=batch_size,scroll="20m", request_timeout=3000)
print("%d documents found" % res['hits']['total'])
print(type(res))
total_size = 0
num_scroll = MAX_SCROLL/batch_size
count = 0
# with open("data/social.js","w") as f:
#     json.dump(res, f)
with open("data/trainprocess.txt", "r") as f:
    string = f.read()
with open("data/data_social_2.txt", "w") as f:
    f.write(string)
    sid = res['_scroll_id']
    scroll_size = res['hits']['total']
    hits_size = len(res['hits']['hits'])

    # Start scrolling
    while (total_size < MAX_SCROLL):
        count +=1
        print("Scrolling...%d/%d" %(count, num_scroll))
        res = es.scroll(scroll_id=sid, scroll='2m')
        # Update the scroll ID
        sid = res['_scroll_id']
        # Get the number of results that we returned in the last scroll
        scroll_size = len(res['hits']['hits'])
        total_size += scroll_size
        print("scroll size: " + str(scroll_size))
        for doc in res['hits']['hits']:
            # print("%s) %s" % (doc['_id'], doc['_source']['content']))
            f.write(doc['_source']['content'])

