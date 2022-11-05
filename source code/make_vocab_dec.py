from data import data_loader
import pickle

dl = data_loader()

vocab = dl.make_data_from_pkl('./pickle/vocab.pkl')
vocab_dec = [' ']*len(vocab)
for each in vocab.keys():
    vocab_dec[vocab[each]] = each

with open('./pickle/vocab_dec.pkl', 'wb') as dec_file:
    pickle.dump(vocab_dec, dec_file)