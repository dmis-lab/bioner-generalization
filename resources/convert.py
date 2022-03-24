import os

data = 'BC5CDR-chem'

def convert(data, file_name, save_name):
    new = []
    with open(os.path.join(data, file_name)) as f:
        lines = f.readlines()
        new = [line.split('\t')[0] for line in lines]

    new = '\n'.join(new)
    with open(os.path.join(data, save_name), 'w') as g:
        g.write(new)

convert(data, 'mention_dict_train_raw.txt', 'mention_dictionary.txt')
convert(data, 'cui_dict_train.txt', 'cui_dictionary.txt')

