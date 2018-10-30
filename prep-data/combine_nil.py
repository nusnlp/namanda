import json
import sys
import os
from tqdm import tqdm
from copy import deepcopy

def load_dict(fname):
    with open(fname, 'r') as fp:
        data = json.load(fp)
    return data


def config_data(datain):
    count = 0
    data = deepcopy(datain)
    ctxs = [p['context'] for p in data['data'][0]['paragraphs']]
    qas = [p['qas'][0] for p in data['data'][0]['paragraphs']]
    # qas = sum(qas, [])
    ctx_set = set(ctxs)
    ctx_list = list(ctx_set)
    new_data = {}
    new_data['version'] = data['version']
    # new_data['data'] = []
    # new_data['paragraphs'] = []
    paras = []
    for ctx in ctx_list:
        paradict = {}
        paradict['context'] = ctx
        paradict['qas'] = []
        for i in range(len(ctxs)):
            if ctxs[i] == ctx:
                paradict['qas'].append(qas[i])
                count += 1
        paras.append(paradict)
    new_data['data'] = [{'paragraphs': paras}]
    print("Total number of questions:", count)
    # new_data['paragraphs'] = paras
    return new_data


def combine_data(data_orig, nil_data_orig):
    assert len(data_orig['data']) == 1
    assert len(nil_data_orig['data']) == 1
    data_orig['data'][0]['paragraphs'] += nil_data_orig['data'][0]['paragraphs']
    return data_orig


datao = load_dict(sys.argv[1])
nil_data = load_dict(sys.argv[2])
dump_path = sys.argv[3]

comb_data = combine_data(datao, nil_data)
with open(dump_path, 'w') as fp:
    json.dump(config_data(comb_data), fp)