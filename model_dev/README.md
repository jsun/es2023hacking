```
python make_train_txt.py db.info.gz train.txt
shuf -n 10000 train.txt > train.10k.txt
shuf -n 100 train.txt > valid.100.txt
```

