import os
import pickle
import utils

tag_list = {'O': 0, 'B-geo-loc': 1, 'B-facility': 2, 'I-facility': 3, 'B-movie': 4, 'I-movie': 5, 'B-company': 6,
            'B-product': 7, 'B-person': 8, 'B-other': 9, 'I-other': 10, 'B-sportsteam': 11, 'I-sportsteam': 12,
            'I-product': 13, 'I-company': 14, 'I-person': 15, 'I-geo-loc': 16, 'B-tvshow': 17, 'B-musicartist': 18,
            'I-musicartist': 19, 'I-tvshow': 20}

data_file = './dataset/train.txt'
split_char = '\t'

if 'train.pkl' in os.listdir('./dataset'):
    with open('./dataset/train.pkl', 'rb') as file_read:
        data = pickle.load(file_read)
else:
    data = {'data': [], 'label': []}

with open(data_file, encoding='utf8') as file:
    sentence = []
    label = []
    for line in file.readlines():
        if line != '\n':
            word = line[:-1].split(split_char)
            if ('@' not in word[0]) and ('/' not in word[0]) and (len(word[0]) > 0):
                sentence.append(word[0])
                label.append(word[-1])
        else:
            if len(sentence) > 4:
                data['data'].append(sentence)
                data['label'].append(label)
            sentence = []
            label = []

    file_write = open('./dataset/train.pkl', 'wb')
    pickle.dump(data, file_write)
    file_write.close()
