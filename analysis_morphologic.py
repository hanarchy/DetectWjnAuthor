from get_wjn_content import WjnContents
import numpy as np
from janome.tokenizer import Tokenizer
from tqdm import tqdm
import os.path
import pickle


pkl_word_dict = 'pkl/word_dict.pkl'


def dict_word(word, p_dict, label):
    if len(word) > 1:
        if p_dict == []:
            if label == 1:
                p_dict.append([word, 1, 1, 0])
            else:
                p_dict.append([word, 1, 0, 1])
        else:
            p_dict_word = [i[0] for i in p_dict]
            if word in p_dict_word:
                index_word = p_dict_word.index(word)
                p_dict[index_word][1] += 1
                if label == 1:
                    p_dict[index_word][2] += 1
                else:
                    p_dict[index_word][3] += 1

            else:
                if label == 1:
                    p_dict.append([word, 1, 1, 0])
                else:
                    p_dict.append([word, 1, 0, 1])
    return p_dict


def create_word_dict(contents, labels):
    all_dict = []
    available_norm = ['接尾', '一般', '形容動詞語幹', 'サ変接続']
    for i in tqdm(range(len(contents))):
        t = Tokenizer()
        for token in t.tokenize(contents[i]):
            pos = token.part_of_speech.split(',')
            if pos[0] == '名詞':
                if pos[1] in available_norm:
                    dict_word(token.surface, all_dict, labels[i])
            if pos[0] == '動詞':
                dict_word(token.surface, all_dict, labels[i])
            if pos[0] == '形容詞':
                dict_word(token.surface, all_dict, labels[i])

    sorted_all_dict = sorted(all_dict, key=lambda x: int(x[1]))

    sorted_all_dict = [i for i in sorted_all_dict if i[1] >= 35]

    sorted_all_dict.reverse()

    num_dict = [i[1] for i in sorted_all_dict]
    word_dict = [i[0] for i in sorted_all_dict
                 if abs(i[2]-i[3])/i[1] >= 0.2]

    print('length of word_dict: %d' % len(word_dict))
    with open(pkl_word_dict, 'wb') as f:
        pickle.dump(word_dict, f)

    return word_dict


def create_datasets(contents, labels=None, create_dict=False):
    if create_dict:
        word_dict = create_word_dict(contents, labels)
    else:
        with open(pkl_word_dict, 'rb') as f:
            word_dict = pickle.load(f)
    # print(len(contents))
    dicts = np.zeros((len(contents), len(word_dict)))
    arr_labels = np.zeros((len(contents), 2))
    for i in range(len(contents)):
        t = Tokenizer()
        for token in t.tokenize(contents[i]):
            if token.surface in word_dict:
                dicts[i][word_dict.index(token.surface)] = 1
        if labels:
            if int(labels[i]) == 1:
                arr_labels[i][0] = 1
            else:
                arr_labels[i][1] = 1
    if labels:
        return dicts, arr_labels
    else:
        return dicts
