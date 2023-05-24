import os
import sys
import gzip


def make_train_txt(input_fpath, output_fpath):

    # open file and load meta data
    meta_infos = []
    with gzip.open(input_fpath, 'rt') as f:
        f.readline()
        for line in f:
                line = line.rstrip()
                words = line.split('\t')
                img_fpath = os.path.join('/data/workshop1/pdbidb_1k',
                                         words[1], words[2], words[3], words[0])
                img_label = words[1]
                img_groupkey = '{}_{}_{}'.format(words[1], words[2], words[8])
                meta_infos.append([img_fpath, img_label, img_groupkey])

    # sort list by the 3th column (i.e., plant_part_datetime)
    meta_infos.sort(key=lambda x: x[2])

    # make a dictionary to store the meta data, grouped by plant_part_datetime
    meta_dict = {}
    for meta_info in meta_infos:
        k = meta_info[2][:-2]
        if k not in meta_dict:
            meta_dict[k] = []
        meta_dict[k].append(meta_info)

    # select one image from meta_dict for each key
    nr_meta_dict = {}
    for k, v in meta_dict.items():
        if len(v) > 1:
            nr_meta_dict[k] = v[0]

    # save image path and label to output_fpath
    with open(output_fpath, 'w') as g:
        for k, v in nr_meta_dict.items():
            g.write('{}\t{}\n'.format(v[0], v[1]))



if __name__ == '__main__':
    make_train_txt(sys.argv[1], sys.argv[2])
