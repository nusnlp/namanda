import sys
import json
import newsqa_evaluator as Evaluator


def load_dict(fname):
    with open(fname, 'r') as fp:
        data = json.load(fp)
    return data


dataset = load_dict('data/datasets/newsqa/newsqa-combined-orig-nil-data/test_nil.json.prepend_nil')
orig_dataset = load_dict('data/datasets/newsqa/NewsQA-v1.1-test.json.configd')

pred = load_dict(sys.argv[1])
for key in pred.keys():
    if 'NIL' in pred[key]:
        pred[key] = "NIL"

print "=" * 50
print "On non-nil data (w/o NIL)"
print(Evaluator.evaluate(orig_dataset['data'], pred))
print "=" * 50

corr_nil = 0.0
pred_nil = 0.0
total_nil = 0

for key in pred.keys():
    if 'NONE' in key:
        total_nil += 1
        if 'NIL' in pred[key]:
            corr_nil += 1.0
    if 'NIL' in pred[key]:
        pred_nil += 1.0

nil_p = corr_nil / pred_nil
nil_r = corr_nil / total_nil
print "NIL prec:", corr_nil / pred_nil
print "NIL recl:", corr_nil / total_nil
print "NIL F1:", 2 * nil_p * nil_r / (nil_p + nil_r)
print "=" * 50
print "Overall scores:"
print(Evaluator.evaluate(dataset['data'], pred))
print "=" * 50
