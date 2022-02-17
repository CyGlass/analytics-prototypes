from __future__ import division
import pandas as pd
from elasticsearch.helpers import scan
from cluster_common import get_escl
import numpy as np
from datetime import datetime


def get_user_docs(escl, raw_login_index, user_id, rem_locality, rem_country):
    """
    Returns raw logins for a given user, locality, country pair.

    :param escl:
    :type escl:
    :param raw_login_index:
    :type raw_login_index:
    :param user_id:
    :type user_id:
    :param rem_locality:
    :type rem_locality:
    :param rem_country:
    :type rem_country:
    :return:
    :rtype:
    """
    user_docs = []
    search_param = {
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "user_id": {
                                "value": user_id
                            }
                        }
                    },
                    {
                        "term": {
                            "rem_locality": {
                                "value": rem_locality
                            }
                        }
                    },
                    {
                        "term": {
                            "rem_country": {
                                "value": rem_country
                            }
                        }
                    },
                    {
                        "term": {
                            "operation": {
                                "value": "UserLoggedIn"
                            }
                        }
                    }
                ]
            }
        }
    }
    for doc in scan(client=escl, index=raw_login_index, query=search_param):
        user_doc = {
            'user_id': user_id,
            'rem_locality': rem_locality,
            'rem_country': rem_country,
            'rem_latitude': doc['_source']['rem_latitude'],
            'rem_longitude': doc['_source']['rem_longitude'],
            'timestamp': doc['_source']['timestamp'],
            'datetime': datetime.fromtimestamp(doc['_source']['timestamp'] / 1000),
            'operation': str(doc['_source']['operation'])
        }
        user_docs.append(user_doc)
    print('Length of user_docs: ', len(user_docs))
    return user_docs


def get_model_docs(escl, gmm_index):
    """
    Returns GMM models that are in FITTED or EXPIRED state.

    :param escl:
    :type escl:
    :param gmm_index:
    :type gmm_index:
    :return:
    :rtype:
    """
    gmm_docs = []
    search_param = {
        'query': {'match': {'doc_type': 'models'}}}

    for doc in scan(client=escl, index=gmm_index, query=search_param):
        if not doc['_source']['state'] == 'COLLECTING':
            gmm_doc = {
                'model_id': str(doc['_source']['model_id']),
                'start_time': doc['_source']['start_time'],
                'end_time': doc['_source']['end_time'],
                'state': str(doc['_source']['state']),
                'means': doc['_source']['GMM_params']['means'],
                'feat_scales': doc['_source']['GMM_params']['feat_scales'],
                'feat_mins': doc['_source']['GMM_params']['feat_mins'],
                'covariances': doc['_source']['GMM_params']['covariances']
            }
            gmm_docs.append(gmm_doc)
    print('Detected: ', len(gmm_docs), 'gmm models')
    return gmm_docs


def unscale_gmm_docs(gmm_docs):
    """
    GMM docs has scaled values for means, lat, lon.

    Returns unscaled (original) version of the GMM docs.

    :param gmm_docs:
    :type gmm_docs:
    :return:
    :rtype:
    """
    unscaled_gmm_docs = []
    for idx, doc in enumerate(gmm_docs):
        unscaled_doc = {
            'model_id': doc['model_id'],
            'start_time': doc['start_time'],
            'end_time': doc['end_time'],
            'state': doc['state']
        }
        unscaled_means = (np.array(doc['means']) - np.array(doc['feat_mins'])) / np.array(doc['feat_scales'])
        unscaled_means_lat_lon = [[i[2], i[3]] for i in list(unscaled_means)]
        unscaled_doc['unscaled_means_lat_lon'] = unscaled_means_lat_lon
        # unscaled_clust_feat_vars_K = GMM_model.covariances_[:, f_idx, f_idx] / math.pow(feat_scales[f_idx], 2)
        unscaled_stds_all = []
        for centroid in doc['covariances']:
            unscaled_var = np.array(centroid).diagonal() / (np.array(doc['feat_scales']) ** 2)
            unscaled_std = unscaled_var ** 0.5
            unscaled_stds_all.append(unscaled_std)
        unscaled_stds_lat_lon = [[i[2], i[3]] for i in unscaled_stds_all]
        unscaled_doc['unscaled_stds_lat_lon'] = unscaled_stds_lat_lon
        unscaled_gmm_docs.append(unscaled_doc)
    return unscaled_gmm_docs


def get_cluster_centroids(unscaled_gmm_docs):
    """
    Returns centroids from each GMM model. One GMM model cam have multiple centroids (clusters).

    :param unscaled_gmm_docs:
    :type unscaled_gmm_docs:
    :return:
    :rtype:
    """
    cluster_centroids = []
    for doc in unscaled_gmm_docs:
        for i in range(len(doc['unscaled_means_lat_lon'])):
            centroid = {'model_id': doc['model_id'], 'start_time': doc['start_time'], 'end_time': doc['end_time'],
                        'state': doc['state'], 'lat_mean': doc['unscaled_means_lat_lon'][i][0],
                        'lon_mean': doc['unscaled_means_lat_lon'][i][1], 'lat_std': doc['unscaled_stds_lat_lon'][i][0],
                        'lon_std': doc['unscaled_stds_lat_lon'][i][1]}
            cluster_centroids.append(centroid)
    return cluster_centroids


def get_z_score(user_doc, cluster_centroids):
    """
    Calculates Z score for a given user login doc.

    :param user_doc:
    :type user_doc:
    :param cluster_centroids:
    :type cluster_centroids:
    :return:
    :rtype:
    """
    user_z_scores = []
    for cluster in cluster_centroids:
        lat_z_score = abs(user_doc['rem_latitude'] - cluster['lat_mean']) / cluster['lat_std']
        lon_z_score = abs(user_doc['rem_longitude'] - cluster['lon_mean']) / cluster['lon_std']
        temp = [cluster['model_id'],
                cluster['start_time'],
                cluster['end_time'],
                cluster['state'],
                cluster['lat_mean'],
                cluster['lon_mean'],
                lat_z_score + lon_z_score
                ]
        user_z_scores.append(temp)
    return sorted(user_z_scores, key=lambda x: x[-1])


def main():
    # Arguments
    user_id = "david.joy@birmingham.gov.uk"
    rem_locality = "New York City"  # Raw login from rawappidaccmgt
    rem_country = "United States of America"
    SRC_SITE = 'bcc'
    RAW_LOGIN_INDEX = "rawappidaccmgt"
    GMM_INDEX = "saas_gmm_table"

    src_url = "https://cyglass:cyglass@" + SRC_SITE + ".cyglass.com:9200/"
    src_escl = get_escl(src_url)
    user_docs = get_user_docs(src_escl, RAW_LOGIN_INDEX, user_id, rem_locality, rem_country)
    gmm_docs = get_model_docs(src_escl, GMM_INDEX)
    unscaled_docs = unscale_gmm_docs(gmm_docs)
    cluster_centroids = get_cluster_centroids(unscaled_docs)
    user_z_scores = get_z_score(user_docs[0], cluster_centroids)
    df = pd.DataFrame(user_z_scores)
    pd.set_option('display.max_columns', None)
    df.columns = ["model", 'start_time', 'end_time', 'status', 'centroid_lat', 'centroid_lon', 'z_score']
    print(df.sort_values('z_score', ascending=False).reset_index())


if __name__ == "__main__":
    main()
