from __future__ import division
from elasticsearch.helpers import scan, parallel_bulk
from cluster_common import get_escl, get_user_location_schema, create_index, generator_docs
import numpy as np
from collections import deque


def scan_gmm_models(escl, index):
    """
    Retrieves GMM models.

    :param escl:
    :type escl:
    :param index:
    :type index:
    :return:
    :rtype:
    """
    # don;t populate from ml event, populate baselines from here. 
    gmm_docs = []
    search_param = {
      "query": {
        "term": {
          "doc_type": {
            "value": "models"
          }
        }
      }
    }

    for doc in scan(client=escl, index=index, query=search_param):
        if not doc['_source']['state'] == 'COLLECTING':
            gmm_doc = {
                'model_id': str(doc['_source']['model_id']),
                'start_time': doc['_source']['start_time'],
                'end_time': doc['_source']['end_time'],
                'state': str(doc['_source']['state']),
                'means': doc['_source']['GMM_params']['means'],
                'feat_scales': doc['_source']['GMM_params']['feat_scales'],
                'feat_mins': doc['_source']['GMM_params']['feat_mins']
            }
            gmm_docs.append(gmm_doc)
    print('Detected: ', len(gmm_docs), ' gmm models')
    return gmm_docs


def unscale_mean(gmm_docs):
    """
    GMM centroids(clusters) are scaled for lat, lon values.

    Return unscaled(original) version of  lat, lon for centroids.

    :param gmm_docs:
    :type gmm_docs:
    :return:
    :rtype:
    """
    # unscaled_clust_feat_means_K = (GMM_model.means_[:, f_idx] - feat_mins[f_idx]) / feat_scales[f_idx]
    gmm_docs_unscaled = gmm_docs[:]
    for count, doc in enumerate(gmm_docs):
        unscaled_means = (np.array(doc['means']) - np.array(doc['feat_mins'])) / np.array(doc['feat_scales'])
        unscaled_lat_lon = [[i[2], i[3]] for i in list(unscaled_means)]
        gmm_docs_unscaled[count]['unscaled_lat_lon'] = unscaled_lat_lon
    return gmm_docs_unscaled


def format_gmm(unscaled_gmm_docs, src_site):
    """
    Formats docs, prepares for Kibana visualization.

    :param unscaled_gmm_docs:
    :type unscaled_gmm_docs:
    :param src_site:
    :type src_site:
    :return:
    :rtype:
    """
    formatted_unscaled_gmm_docs = []
    for doc in unscaled_gmm_docs:
        for centroid in doc['unscaled_lat_lon']:
            baseline_loc = {}
            baseline_loc['user_id'] = 'model_baseline'
            baseline_loc['location_type'] = 'baseline'
            baseline_loc['location'] = {'lat': centroid[0],
                                        'lon': centroid[1]}
            baseline_loc['lat_diff'] = 0.0
            baseline_loc['lon_diff'] = 0.0
            baseline_loc['site_name'] = src_site
            formatted_unscaled_gmm_docs.append(baseline_loc)
    print('Total formatted unscaled gmm docs: ', len(formatted_unscaled_gmm_docs))
    return formatted_unscaled_gmm_docs


def main():
    # Arguments
    persist = True
    SRC_SITE = "bcc"
    DST_SITE = "devbcc6"
    GMM_INDEX = "saas_gmm_table"
    TARGET_INDEX = "site_model_clusters_test"

    # SRC Data Retrieval
    src_url = "https://cyglass:cyglass@" + SRC_SITE + ".cyglass.com:9200/"
    src_escl = get_escl(src_url)
    gmm_docs = scan_gmm_models(src_escl, GMM_INDEX)
    unscaled_gmm_docs = unscale_mean(gmm_docs)
    formatted_unscaled_gmm_docs = format_gmm(unscaled_gmm_docs, SRC_SITE)

    # Write to ES
    if persist:
        dst_url = "https://cyglass:cyglass@" + DST_SITE + ".cyglass.com:9200/"
        dst_escl = get_escl(dst_url)
        schema = get_user_location_schema()
        create_index(dst_escl, set_index=TARGET_INDEX, set_schema=schema)
        deque(parallel_bulk(dst_escl, generator_docs(formatted_unscaled_gmm_docs, TARGET_INDEX)), maxlen=0)


if __name__ == "__main__":
    main()
