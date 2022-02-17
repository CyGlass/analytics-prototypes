from __future__ import division, print_function
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk, parallel_bulk
import json
from collections import deque

SRC_SITE = "devfonex1"
DST_SITE = "devfonex1"
DOC_TYPE = "cyglass"
MAP_ML_EVENT_INDEX_ = "ml_event"
MAP_RAW_LOACATION_INDEX = "rawappidaccmgt"
TARGET_INDEX = "user_location_test2"


def get_escl(url):
    '''
    Creates Elastic Search instance and connects to url.
    Ruturns escl object if there is a connection, otherwise returns False.
    '''
    escl = Elasticsearch(hosts=url, timeout=300)
    if escl.ping():
        return escl
    return False


def create_index(escl, set_index, set_schema):
    """
    Setup  Elastic Search indices for test data
    :param set_index:
    :param set_schema:
    :return:
    """
    # create index in ES if indices do not exist
    if not escl.indices.exists(index=set_index):
        print("Detected no index by the name %s in Elasticsearch" % set_index)
        try:
            escl.indices.create(index=set_index, body=set_schema, ignore=400)
            print("Created ES index %s ... " % set_index)
        except Exception:
            print("Failed to create ES index %s ... " % set_index)
        print("Created schema for %s in Elasticsearch" % set_index)


def loop_index_to_escl(escl, index, docs, doc_type=DOC_TYPE):
    '''
    Writes list of docs to Elastic Search.
    Return 
    '''
    if docs:
        for count, doc in enumerate(docs):
            try:
                escl.index(index=index, doc_type=doc_type, body=doc)
                if len(docs) >= 1000:
                    if count % 1000 == 0:
                        print("Percentage completed: ", count/len(docs) * 100)
                else:
                    print("Percentage completed: ", count/len(docs) * 100)
            except Exception:
                print("Failed to index to", index)
                print("Check ES instance or index")
    print("Completed writing to index: ", index)


def scan_anomaly_locations(escl, index):
    '''
    Takes ESCL intsance
    Return list of docs(type: dict) of anomaly loactions.
    '''
    anomaly_docs = []
    user_ids = set()
    search_param = {
        'query': {'match': {'anomtype': 'Unusual Access Location For a User'}}}
    for doc in scan(client=escl, index=index, query=search_param):
        target_doc = {'user_id': str(doc['_source']['endpoints'][0]['value']),
                      'location_type': 'anomaly',
                      'location': {'lat': doc['_source']['triggering_features_by_model']['main_triggering_feature']['value_location']['lat'],
                                   'lon': doc['_source']['triggering_features_by_model']['main_triggering_feature']['value_location']['lon']},
                      'lat_z_score': abs(doc['_source']['triggering_features_by_model']['main_triggering_feature']['value_location']['lat'] -
                                         doc['_source']['triggering_features_by_model']['main_triggering_feature']['baseline_location']['lat']) / doc['_source']['triggering_features_by_model']['main_triggering_feature']['standard_deviation'][0],
                      'lon_z_score': abs(doc['_source']['triggering_features_by_model']['main_triggering_feature']['value_location']['lon'] -
                                         doc['_source']['triggering_features_by_model']['main_triggering_feature']['baseline_location']['lon']) / doc['_source']['triggering_features_by_model']['main_triggering_feature']['standard_deviation'][1],
                      'site_name': SRC_SITE}

        anomaly_docs.append(target_doc)
        user_ids.add(target_doc['user_id'])

    print('Detected ', len(anomaly_docs), 'anomaly location events in',SRC_SITE,
          ' generated by', len(user_ids), 'unique users')
    return anomaly_docs


def scan_raw_locations(escl, raw_loc_index):
    '''
    Take unique user_ids set
    Return Row Location Event docs
    '''
    raw_loc_docs = []
    raw_loc_search_param = {"query": {"bool": {"must": [{"match_all": {}}]}}}
    for doc in scan(client=escl, index=raw_loc_index, query=raw_loc_search_param):
        loc_type = ''
        if doc['_source']['operation'] == 'UserLoggedIn':
            loc_type = 'successfulLoginLocation'
        elif doc['_source']['operation'] == 'UserLoginFailed':
            loc_type = 'failedLoginLocation'
        else:
            continue
        raw_loc_doc = {'user_id': doc['_source']['user_id'],
                                'location_type': loc_type,
                                'location': {'lat': doc['_source']["rem_latitude"],
                                            'lon': doc['_source']["rem_longitude"]
                                            },
                                'lat_diff': 0.0, 
                                'lon_diff': 0.0,
                                'site_name': SRC_SITE}
        raw_loc_docs.append(raw_loc_doc)
    print('Detected ', len(raw_loc_docs), 'raw location events')
    return raw_loc_docs


def get_user_location_schema():
    '''
    Return shcema for user_location
    '''
    schema = {"mappings": {
        "cyglass": {
            "properties": {
                "area": {"type": "geo_shape"},
                "user_id": {"type": "keyword"},
                "location": {"type": "geo_point"},
                "location_type": {"type": "keyword"},
                "lat_z_score": {"type": "double"},
                "lon_z_score": {"type": "double"},
                "site_name": {"type": "keyword"}
            }
        }
    }
    }
    return schema

def generator_docs(docs, index, doc_type):
    for doc in docs:
        doc['_index'] = index
        doc['_type'] = doc_type
        yield doc

def main():
    ### SRC Site Data Retrieval
    src_url = "https://cyglass:cyglass@"+SRC_SITE+".cyglass.com:9200/"
    src_escl = get_escl(src_url)
    anomaly_docs = scan_anomaly_locations(src_escl, MAP_ML_EVENT_INDEX_)

    ### DST Site Data Upload (Anomaly locations)
    dst_url = "https://cyglass:cyglass@"+DST_SITE+".cyglass.com:9200/"
    dst_escl = get_escl(dst_url)
    if not dst_escl.indices.exists(index=TARGET_INDEX):
        user_location_schema = get_user_location_schema()
        create_index(dst_escl, set_index=TARGET_INDEX,
                     set_schema=user_location_schema)
        print('Cretaed index: ', TARGET_INDEX, 'in ', DST_SITE)
    else:
        print(TARGET_INDEX, 'is already in',
              DST_SITE, 'no need to create index')

    #???? IF I re run this command it writes duplicates, is there any duplicate prevention solution ? (UUID or hash)
    # loop_index_to_escl(dst_escl, index=TARGET_INDEX, docs=anomaly_docs)
    
    ### DST Site Data Upload (parallel_bulk) Anomaly locations
    deque(parallel_bulk(dst_escl, generator_docs(anomaly_docs)), maxlen=0)

    ### DST Site Data Upload (parallel_bulk) Raw locations
    raw_docs = scan_raw_locations(src_escl, MAP_RAW_LOACATION_INDEX) 
    # # loop_index_to_escl(dst_escl, index=TARGET_INDEX, docs=raw_docs)
    deque(parallel_bulk(dst_escl, generator_docs(raw_docs), 10), maxlen=0)

if __name__ == "__main__":
    main()