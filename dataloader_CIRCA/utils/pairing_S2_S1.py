import datetime
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_datetime(date):
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])
    return datetime.datetime(year, month, day)


def convert_dates(dates):
    dates_list = []
    d0 = datetime.datetime(2022, 9, 1)
    for date in dates:
        ecart = get_datetime(date) - d0
        dates_list.append(ecart.days)
    return np.array(dates_list)


def appariement_S1_to_S2(
    S2_dates,
    S1_dates_asc,
    S1_dates_desc,
):
    S1_dates_asc_int = convert_dates(S1_dates_asc)
    S1_dates_desc_int = convert_dates(S1_dates_desc)
    S2_dates_int = convert_dates(S2_dates)
    S1_dates_int = np.concatenate([S1_dates_asc_int, S1_dates_desc_int])
    indices = np.argsort(S1_dates_int)
    S1_dates_int = S1_dates_int[indices]
    S1_dates_tmp = S1_dates_asc + S1_dates_desc
    S1_dates = np.array([S1_dates_tmp[indices[i]] for i in range(indices.shape[0])])

    # Détermination si la donnée SAR la plus proche temporellement est ASC ou DESC
    dict_appariement_S2_S1_asc = {}
    list_index_prelevement_asc = []
    distance_S2_S1_asc = np.abs(np.expand_dims(S2_dates_int, 1) - np.expand_dims(S1_dates_asc_int, 0))
    id_min_S2_S1_asc = np.argmin(distance_S2_S1_asc, axis=1)
    for i in range(len(S2_dates)):
        date_min = S1_dates_asc[id_min_S2_S1_asc[i]]
        orbit_type = "ASC"
        index_prelevement_asc = S1_dates_asc.index(date_min)
        if index_prelevement_asc not in list_index_prelevement_asc:
            list_index_prelevement_asc.append(index_prelevement_asc)
        index_in_new_data_asc = list_index_prelevement_asc.index(index_prelevement_asc)
        dict_appariement_S2_S1_asc[S2_dates[i]] = (date_min, orbit_type, index_in_new_data_asc)

    dict_appariement_S2_S1_desc = {}
    list_index_prelevement_desc = []
    distance_S2_S1_desc = np.abs(np.expand_dims(S2_dates_int, 1) - np.expand_dims(S1_dates_desc_int, 0))
    id_min_S2_S1_desc = np.argmin(distance_S2_S1_desc, axis=1)
    for i in range(len(S2_dates)):
        date_min = S1_dates_desc[id_min_S2_S1_desc[i]]
        orbit_type = "DESC"
        index_prelevement_desc = S1_dates_desc.index(date_min)
        if index_prelevement_desc not in list_index_prelevement_desc:
            list_index_prelevement_desc.append(index_prelevement_desc)
        index_in_new_data_desc = list_index_prelevement_desc.index(index_prelevement_desc)
        dict_appariement_S2_S1_desc[S2_dates[i]] = (date_min, orbit_type, index_in_new_data_desc)

    dates_S1_ASC_collected = np.asarray(S1_dates_asc)[list_index_prelevement_asc].tolist()
    dates_S1_DESC_collected = np.asarray(S1_dates_desc)[list_index_prelevement_desc].tolist()

    # Détermination si la donnée SAR la plus proche temporellement est ASC ou DESC
    distance_S2_S1_all = np.abs(np.expand_dims(S2_dates_int, 1) - np.expand_dims(S1_dates_int, 0))
    id_min_S2_S1_all = np.argmin(distance_S2_S1_all, axis=1)
    dict_appariement_S2_S1_all = {}
    for i in range(len(S2_dates)):
        date_min = S1_dates[id_min_S2_S1_all[i]]
        if date_min in S1_dates_asc:
            orbit_type = "ASC"
            index = dates_S1_ASC_collected.index(date_min)
        else:
            orbit_type = "DESC"
            index = dates_S1_DESC_collected.index(date_min)
        dict_appariement_S2_S1_all[S2_dates[i]] = (date_min, orbit_type, index)

    dict_appariement = {
        "asc": dict_appariement_S2_S1_asc,
        "desc": dict_appariement_S2_S1_desc,
        "mix_closest": dict_appariement_S2_S1_all,
    }

    assert len(list_index_prelevement_asc) == len(dates_S1_ASC_collected)
    assert len(list_index_prelevement_desc) == len(dates_S1_DESC_collected)

    return (
        list_index_prelevement_asc,
        list_index_prelevement_desc,
        dates_S1_ASC_collected,
        dates_S1_DESC_collected,
        dict_appariement,
    )
