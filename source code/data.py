import pickle

class data_loader:
    def __init__(self):
        pass

    def make_data_from_pkl(self, file_dir):
        data_file = open(file_dir, 'rb')
        data = pickle.load(data_file)
        data_file.close()
        return data