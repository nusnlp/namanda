import json
import sys
import os
import nltk
from tqdm import tqdm


def get_tripped_sample(data, maxctxlen):
    new_data = {}
    new_data['id'] = data['id']
    new_data['question'] = data['question']
    new_data['qlemma'] = data['qlemma']

    story_words = data['document']
    story_offsets = data['offsets']
    story_pos = data['pos']
    story_lemma = data['lemma']
    story_ner = data['ner']
    answers = data['answers']

    if len(story_words) > maxctxlen:
        start_ind = answers[0][0]
        alen = answers[0][1] - start_ind + 1
        if start_ind + alen <= maxctxlen:
            story_words = story_words[:maxctxlen]
            story_offsets = story_offsets[:maxctxlen]
            story_pos = story_pos[:maxctxlen]
            story_lemma = story_lemma[:maxctxlen]
            story_ner = story_ner[:maxctxlen]
        elif start_ind + alen > maxctxlen / 2 and len(story_words) - (start_ind + alen) > maxctxlen / 2:
            astart = [0] * len(story_words)
            astart[start_ind] = 1
            story_words = story_words[start_ind - maxctxlen / 2: start_ind] + \
                          story_words[start_ind: start_ind + alen] + \
                          story_words[start_ind + alen: start_ind + alen + maxctxlen / 2]
            story_pos = story_pos[start_ind - maxctxlen / 2: start_ind] + \
                        story_pos[start_ind: start_ind + alen] + \
                        story_pos[start_ind + alen: start_ind + alen + maxctxlen / 2]
            story_offsets = story_offsets[start_ind - maxctxlen / 2: start_ind] + \
                            story_offsets[start_ind: start_ind + alen] + \
                            story_offsets[start_ind + alen: start_ind + alen + maxctxlen / 2]
            story_lemma = story_lemma[start_ind - maxctxlen / 2: start_ind] + \
                          story_lemma[start_ind: start_ind + alen] + \
                          story_lemma[start_ind + alen: start_ind + alen + maxctxlen / 2]
            story_ner = story_ner[start_ind - maxctxlen / 2: start_ind] + \
                        story_ner[start_ind: start_ind + alen] + \
                        story_ner[start_ind + alen: start_ind + alen + maxctxlen / 2]
            astart = astart[start_ind - maxctxlen / 2: start_ind] + \
                     astart[start_ind: start_ind + alen] + \
                     astart[start_ind + alen: start_ind + alen + maxctxlen / 2]
            answers = [[astart.index(1), astart.index(1) + alen - 1]]
        elif start_ind + alen > maxctxlen / 2 and len(story_words) - (start_ind + alen) <= maxctxlen / 2:
            astart = [0] * len(story_words)
            astart[start_ind] = 1
            story_words = story_words[
                          max(0, start_ind - (maxctxlen - (len(story_words) - start_ind - alen))):start_ind] + \
                          story_words[start_ind:]
            story_pos = story_pos[max(0, start_ind - (
                maxctxlen - (len(story_pos) - start_ind - alen))):start_ind] + \
                        story_pos[start_ind:]
            story_ner = story_ner[max(0, start_ind - (
                maxctxlen - (len(story_ner) - start_ind - alen))):start_ind] + \
                        story_ner[start_ind:]
            story_offsets = story_offsets[max(0, start_ind - (
                maxctxlen - (len(story_offsets) - start_ind - alen))):start_ind] + \
                            story_offsets[start_ind:]
            story_lemma = story_lemma[max(0, start_ind - (
                maxctxlen - (len(story_lemma) - start_ind - alen))):start_ind] + \
                          story_lemma[start_ind:]
            astart = astart[max(0, start_ind - (maxctxlen - (len(astart) - start_ind - alen))):start_ind] + \
                     astart[start_ind:]
            answers = [[astart.index(1), astart.index(1) + alen - 1]]
    new_data['document'] = story_words
    new_data['offsets'] = story_offsets
    new_data['pos'] = story_pos
    new_data['lemma'] = story_lemma
    new_data['ner'] = story_ner
    new_data['answers'] = answers

    return new_data


if __name__ == '__main__':
    maxctxlen = 400
    fname = sys.argv[1]
    data = []
    print('Loading data')
    with open(fname, 'r') as fp:
        for line in fp:
            data.append(json.loads(line))

    print('Tripping')
    with open(fname + '.tripped', 'w') as fp:
        for item in tqdm(data):
            new_data = get_tripped_sample(item, maxctxlen)
            fp.write(json.dumps(new_data) + '\n')
