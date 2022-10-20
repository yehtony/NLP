train_dir_0 = './dataset/0/train.txt'
train_dir_1 = './dataset/1/train.txt'

tag_list = {'O': 0, 'B-geo-loc': 1, 'B-facility': 2, 'I-facility': 3, 'B-movie': 4, 'I-movie': 5, 'B-company': 6,
            'B-product': 7, 'B-person': 8, 'B-other': 9, 'I-other': 10, 'B-sportsteam': 11, 'I-sportsteam': 12,
            'I-product': 13, 'I-company': 14, 'I-person': 15, 'I-geo-loc': 16, 'B-tvshow': 17, 'B-musicartist': 18,
            'I-musicartist': 19, 'I-tvshow': 20, '<unk>': 21, '<pad>': 22, '<start>': 23}

tag_list_rev = ['O', 'B-geo-loc', 'B-facility', 'I-facility', 'B-movie', 'I-movie', 'B-company',
                'B-product', 'B-person', 'B-other', 'I-other', 'B-sportsteam', 'I-sportsteam',
                'I-product', 'I-company', 'I-person', 'I-geo-loc', 'B-tvshow', 'B-musicartist',
                'I-musicartist', 'I-tvshow', '<unk>', '<pad>', '<start>']

extra_tag = {'O': 'O', 'B-ORG': 'B-company', 'B-MISC': 'B-geo-loc', 'B-PER': 'B-person', 'I-PER': 'I-person',
             'B-LOC': 'B-geo-loc', 'I-ORG': 'I-company', 'I-MISC': 'I-geo-loc', 'I-LOC': 'I-geo-loc'}

create_new_model = True
epoch = 80
emb_size = 256
hidden_size = 256
batch_size = 64
lr = 0.001
print_step = 5

import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensorize(batch, maps, tag):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            if tag:
                batch_tensor[i][j] = maps.get(e, UNK)
            else:
                batch_tensor[i][j] = maps.get(e.lower(), UNK)

    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list
