import os
import pickle

tag_list = {'O': 0, 'B-geo-loc': 1, 'B-facility': 2, 'I-facility': 3, 'B-movie': 4, 'I-movie': 5, 'B-company': 6,
            'B-product': 7, 'B-person': 8, 'B-other': 9, 'I-other': 10, 'B-sportsteam': 11, 'I-sportsteam': 12,
            'I-product': 13, 'I-company': 14, 'I-person': 15, 'I-geo-loc': 16, 'B-tvshow': 17, 'B-musicartist': 18,
            'I-musicartist': 19, 'I-tvshow': 20}

data_file = './dataset/test-submit.txt'
split_char = '\t'

if 'test-submit.pkl' in os.listdir('./pickle'):
    with open('./pickle/test-submit.pkl', 'rb') as file_read:
        data = pickle.load(file_read)
else:
    data = {'data': []}

with open(data_file, encoding='utf8') as file:
    sentence = []
    for line in file.readlines():
        if line != '\n':
            word = line[:-1].split()
            sentence.append(word[0])
        else: 
            data['data'].append(sentence)
            sentence = []
         
    file_write = open('./pickle/test-submit.pkl', 'wb')
    pickle.dump(data, file_write)
    file_write.close()