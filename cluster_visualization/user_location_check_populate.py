from __future__ import division, print_function
from elasticsearch.helpers import scan, parallel_bulk
from cluster_common import get_escl, create_index, generator_docs, get_user_location_schema
from collections import deque


def get_anomalous_docs(escl, site_name):
    """
    Retrieves anom access location logins from ml_event index.

    :param escl:
    :type escl:
    :param site_name:
    :type site_name:
    :return:
    :rtype:
    """
    anomalous_docs = []
    search_param = {"query": {"term": {"anomtype": {"value": "Unusual Access Location For a User"}}}}
    anomalous_locations_index = 'ml_event_v1'

    for doc in scan(client=escl, index=anomalous_locations_index, query=search_param):
        target_doc = {'user_id': str(doc['_source']['endpoints'][0]['value']),
                      'location_type': 'anomaly',
                      'location': {'lat': doc['_source']['triggering_features_by_model']['main_triggering_feature'][
                          'value_location']['lat'],
                                   'lon': doc['_source']['triggering_features_by_model']['main_triggering_feature'][
                                       'value_location']['lon']},
                      'lat_z_score': abs(
                          doc['_source']['triggering_features_by_model']['main_triggering_feature']['value_location'][
                              'lat'] -
                          doc['_source']['triggering_features_by_model']['main_triggering_feature'][
                              'baseline_location']['lat']) /
                                     doc['_source']['triggering_features_by_model']['main_triggering_feature'][
                                         'standard_deviation'][0],
                      'lon_z_score': abs(
                          doc['_source']['triggering_features_by_model']['main_triggering_feature']['value_location'][
                              'lon'] -
                          doc['_source']['triggering_features_by_model']['main_triggering_feature'][
                              'baseline_location']['lon']) /
                                     doc['_source']['triggering_features_by_model']['main_triggering_feature'][
                                         'standard_deviation'][1],
                      'site_name': site_name}

        target_doc_baseline = {'user_id': str(doc['_source']['endpoints'][0]['value']),
                               'location_type': 'baseline',
                               'location': {'lat': doc['_source']['triggering_features_by_model'][
                                   'main_triggering_feature']['baseline_location']['lat'],
                                            'lon': doc['_source']['triggering_features_by_model'][
                                                'main_triggering_feature']['baseline_location']['lon']
                                            },
                               'lat_diff': 0.0,
                               'lon_diff': 0.0,
                               'site_name': site_name}

        anomalous_docs.append(target_doc)
        anomalous_docs.append(target_doc_baseline)
    print("Found anom location docs: ", len(anomalous_docs))
    return anomalous_docs


def get_user_ids_anomalous(anomalous_docs):
    """
    Returns list of unique user ids from anom location docs.

    :param anomalous_docs:
    :type anomalous_docs:
    :return:
    :rtype:
    """
    user_ids = set()
    for doc in anomalous_docs:
        user_ids.add(doc['user_id'])
    return list(user_ids)


def get_normal_docs(escl, user_ids, site_name):
    """
    Return normal location logins for set of user ids.

    :param escl:
    :type escl:
    :param user_ids:
    :type user_ids:
    :param site_name:
    :type site_name:
    :return:
    :rtype:
    """
    normal_loc_index = 'rawappidaccmgt_v1'
    normal_loc_docs = []
    normal_loc_search_param = {
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "operation": {
                                "value": "UserLoggedIn"
                            }
                        }
                    },
                    {
                        "terms": {
                            "user_id": user_ids
                        }
                    }
                ]
            }
        }
    }
    for doc in scan(client=escl, index=normal_loc_index, query=normal_loc_search_param):
        normal_loc_doc = {'user_id': doc['_source']['user_id'],
                          'location_type': 'normal',
                          'location': {'lat': doc['_source']["rem_latitude"],
                                       'lon': doc['_source']["rem_longitude"]
                                       },
                          'lat_diff': 0.0,
                          'lon_diff': 0.0,
                          'site_name': site_name}
        normal_loc_docs.append(normal_loc_doc)
    print("Found normal location docs : ", len(normal_loc_docs))
    return normal_loc_docs


def run():
    # Arguments
    persist = False
    SRC_SITE = 'bcc'
    DST_SITE = 'devbcc6'
    TARGET_INDEX = 'users_location_clusters'

    # Run
    src_url = "https://cyglass:cyglass@" + SRC_SITE + ".cyglass.com:9200/"
    src_escl = get_escl(src_url)
    anom_docs = get_anomalous_docs(src_escl, SRC_SITE)
    anom_user_ids = get_user_ids_anomalous(anom_docs)
    normal_docs = get_normal_docs(src_escl, anom_user_ids, SRC_SITE)

    # Write to ES
    if persist:
        dst_url = "https://cyglass:cyglass@" + DST_SITE + ".cyglass.com:9200/"
        dst_escl = get_escl(dst_url)
        schema = get_user_location_schema()
        create_index(dst_escl, set_index=TARGET_INDEX, set_schema=schema)
        deque(parallel_bulk(dst_escl, generator_docs(anom_docs, TARGET_INDEX)), maxlen=0)
        deque(parallel_bulk(dst_escl, generator_docs(normal_docs, TARGET_INDEX)), maxlen=0)


if __name__ == "__main__":
    run()
