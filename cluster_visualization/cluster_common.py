from __future__ import division, print_function
from elasticsearch import Elasticsearch


def get_escl(url):
    """
    Creates ElasticSearch instance and connects to url.

    :param url:
    :type url:
    :return:
    :rtype:
    """
    escl = Elasticsearch(hosts=url, timeout=300)
    if escl.ping():
        return escl
    return False


def create_index(escl, set_index, set_schema):
    """
    Setups  Elastic Search indices for test data

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


def get_user_location_schema():
    """
    Returns schema used for Kibana visualization.

    :return:
    :rtype:
    """
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


def generator_docs(docs, index, doc_type='cyglass'):
    """
    Generator used for parallel_bulk

    :param docs:
    :type docs:
    :param index:
    :type index:
    :param doc_type:
    :type doc_type:
    :return:
    :rtype:
    """
    for doc in docs:
        doc['_index'] = index
        doc['_type'] = doc_type
        yield doc
