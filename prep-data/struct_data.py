import json
import sys


def config_data(data):
    #
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
        paras.append(paradict)
    new_data['data'] = [{'paragraphs': paras}]
    # new_data['paragraphs'] = paras
    return new_data


def load_dict(fname):
    with open(fname, 'r') as fp:
        data = json.load(fp)
    return data


fname = 'data/datasets/newsqa/NewsQA-v1.1-train.json'
fnamedev = 'data/datasets/newsqa/NewsQA-v1.1-dev.json'
fnametest = 'data/datasets/newsqa/NewsQA-v1.1-test.json'

data = load_dict(fname)
new_dataa = config_data(data)
with open(fname + '.configd', 'w') as fp:
    json.dump(new_dataa, fp)

data = load_dict(fnamedev)
new_dataa = config_data(data)
with open(fnamedev + '.configd', 'w') as fp:
    json.dump(new_dataa, fp)

data = load_dict(fnametest)
new_dataa = config_data(data)
with open(fnametest + '.configd', 'w') as fp:
    json.dump(new_dataa, fp)
