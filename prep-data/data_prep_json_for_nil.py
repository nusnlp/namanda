import json
import re
import pandas as pd
import ast
import sys
import uuid
# story_id,question,answer_char_ranges,is_answer_absent,is_question_bad,validated_answers,story_text

#csv_data_path = 'split-data-nil/nil_test.csv'
csv_data_path = sys.argv[1]

data = pd.read_csv(csv_data_path, encoding='utf-8')

story_ids = data['story_id'].values.tolist()
questions = data['question'].values.tolist()
answer_char_rangess = data['answer_char_ranges'].values.tolist()
is_answer_absents = data['is_answer_absent'].values.tolist()
is_question_bads = data['is_question_bad'].values.tolist()
validated_answerss = data['validated_answers'].values.tolist()
story_texts = data['story_text'].values.tolist()

s_data = dict()
s_data['version'] = 'v-1.1'

article = dict()
article['paragraphs'] = []
for sidx in range(len(story_ids)):
    sid = story_ids[sidx]
    paragraph = dict()
    paragraph['context'] = story_texts[sidx]
    #paragraph['qas'] = []
    qans = dict()
    qans['id'] = str(sidx) + uuid.uuid4().hex + 'NONE'
    qans['question'] = questions[sidx]
    qans['answers'] = [{'answer_start':-1, 'text':'NIL'}]
    qans['answer'] = [{'answer_start':-1, 'text':'NIL'}]
    paragraph['qas'] = [qans]
    article['paragraphs'].append(paragraph)

    if (sidx+1)%1000 == 0:
        print("Done for " + str(sidx+1) + " samples..")

s_data['data'] = [article]

print("Dumping data...")
fname = sys.argv[2]
with open(fname, 'w') as fp:
    json.dump(s_data, fp)
