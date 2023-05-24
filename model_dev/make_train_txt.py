import os
import sys
import gzip


'''Create train.txt file for model training

1. open GZIP compressed file which contains the path, class label, and other information
2. retrive the file path and the class label
3. save the file path and the class label to train.txt
'''
def make_train_txt(input_fpath, output_fpath):
    with gzip.open(input_fpath, 'rt') as f:
        with open(output_fpath, 'w') as g:
            for line in f:
                line = line.rstrip()
                words = line.split('\t')
                g.write('{}\t{}\n'.format(words[0], words[1]))




def make_train_txt2(input_fpath, output_fpath):
    meta_infos = []
    with gzip.open(input_fpath, 'rt') as f:
        for line in f:
                line = line.rstrip()
                words = line.split('\t')
                words.append('{}_{}_{}'.format(words[1], words[2], words[8]))
                meta_infos.append(words)

    # sort list by the 10th column
    meta_infos.sort(key=lambda x: x[9])

    # make a dictionary to store the meta data
    meta_dict = {}
    for meta_info in meta_infos:
        k = meta_info[9][:-2]
        if k not in meta_dict:
            meta_dict[k] = []
        meta_dict[k].append(meta_info)

    nr_meta_dict = {}
    for k, v in meta_dict.items():
        if len(v) > 1:
            nr_meta_dict[k] = v[0]

    with open(output_fpath, 'w') as g:
        for k, v in nr_meta_dict.items():
            g.write('{}\t{}\n'.format(v[0], v[1]))



if __name__ == '__main__':
    make_train_txt2(sys.argv[1], sys.argv[2])
