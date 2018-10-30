import json
import sys


def load_dict(fname):
    with open(fname, 'r') as fp:
        data = json.load(fp)
    return data


fname = sys.argv[1]
data = load_dict(fname)
for a in data['data']:
    for p in a['paragraphs']:
        p['context'] = 'NIL ' + p['context']
        for qa in p['qas']:
            for ans in qa['answers']:
                if ans['text'] == 'NIL' or ans['answer_start'] == -1:
                    ans['answer_start'] = 0
                else:
                    ans['answer_start'] = int(ans['answer_start']) + 4
            for ans in qa['all_answers']:
                if ans['text'] == 'NIL' or ans['answer_start'] == -1:
                    ans['answer_start'] = 0
                else:
                    ans['answer_start'] = int(ans['answer_start']) + 4

with open(fname + '.prepend_nil', 'w') as fp:
    json.dump(data, fp)
