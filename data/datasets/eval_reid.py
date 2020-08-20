# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import json
import os


def process_g_pids(q_pid, g_pid_lists):
    g_pids = []
    for g_pid_list in g_pid_lists:
        if len(g_pid_list) <= 1:
            g_pids.append(g_pid_list[0])
        else:
            if q_pid in g_pid_list:
                g_pids.append(q_pid)
            else:
                g_pids.append(g_pid_list[0])
    return np.array(g_pids)


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):

    # print(list(g_pids))

    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []

    flag = 0

    if not isinstance(g_pids[0], (int, str)):
        list_g_pids = g_pids
        flag = 1

    num_valid_q = 0.  # number of valid query

    q_pid_return = -88

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # print(flag)
        if flag == 1:
            g_pids = process_g_pids(q_pid, list_g_pids)
            matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        q_pid_return = q_pid

    if num_valid_q == 0:
        return -1, -1, q_pid_return
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)

    all_cmc = all_cmc.sum(0) / num_valid_q

    mAP = np.mean(all_AP)

    return all_cmc, mAP, q_pid_return
