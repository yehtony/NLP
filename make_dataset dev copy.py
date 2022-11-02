import pandas as pd
import pickle as pkl
import os
import pickle
import utils

# tag_list = {'O': 0, 'B-geo-loc': 1, 'B-facility': 2, 'I-facility': 3, 'B-movie': 4, 'I-movie': 5, 'B-company': 6,
#             'B-product': 7, 'B-person': 8, 'B-other': 9, 'I-other': 10, 'B-sportsteam': 11, 'I-sportsteam': 12,
#             'I-product': 13, 'I-company': 14, 'I-person': 15, 'I-geo-loc': 16, 'B-tvshow': 17, 'B-musicartist': 18,
#             'I-musicartist': 19, 'I-tvshow': 20}

data_file = './dataset/dev.txt'
split_char = '\t'

if 'dev2.pkl' in os.listdir('./dataset'):
    with open('./dataset/dev2.pkl', 'rb') as file_read:
        data = pickle.load(file_read)
else:
    data = {'Sentence': [], 'Word': [], 'Tag': []}

with open(data_file, encoding='utf8') as file:
    i = 1
    for line in file.readlines():
        if line != '\n':
            word = line[:-1].split(split_char)
            if ('@' not in word[0]) and ('/' not in word[0]) and (len(word[0]) > 0):
                data['Word'].append(word[0])
                data['Tag'].append(word[-1])
                data['Sentence'].append('sentence ' + str(i))
        else:
            i += 1
        
    file_write = open('./dataset/dev2.pkl', 'wb')
    pickle.dump(data, file_write)
    file_write.close()


with open("./dataset/dev2.pkl", "rb") as f:
    object = pkl.load(f)

df = pd.DataFrame(object)
df.to_csv(r'./dataset/dev2.csv')
