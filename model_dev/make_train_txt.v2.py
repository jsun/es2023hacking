import os
import sys
import random
import datetime
import string

# input file paht
in_fpath = sys.argv[1]


dat = {}
with open(in_fpath, 'r') as infh:
    for line in infh:
        line = line.replace('\n', '')
        shoot_dt = line.split('\t')[3]
        if shoot_dt not in dat:
            dat[shoot_dt] = []
        dat[shoot_dt].append(line)



prev_dt = datetime.datetime.strptime('1990:01:01 00:00:00', '%Y:%m:%d %H:%M:%S')
random_key = None

for dtkey in sorted(dat.keys()):
    imageinfos = dat[dtkey]
    if len(imageinfos) > 1:
        print('The following images are taken in the same time! Only the first one will be used,')
        print(imageinfos)

    shoot_dt = imageinfos[0].split('\t')[3]
    shoot_dt = datetime.datetime.strptime(shoot_dt, '%Y:%m:%d %H:%M:%S')
    diff_dt = shoot_dt - prev_dt

    if diff_dt.total_seconds() > 60:
        prev_dt = shoot_dt
        random_key = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        dat[random_key] = []
    dat[random_key].append(imageinfos[0])


out_fpath = os.path.splitext(in_fpath)[0] + '.cv.tsv'
with open(out_fpath, 'w') as outfh:
    for random_key, images in dat.items():
        i = random.randint(0, 9)
        for image in images:
            outfh.write('{}\t{}\n'.format(image, i))




