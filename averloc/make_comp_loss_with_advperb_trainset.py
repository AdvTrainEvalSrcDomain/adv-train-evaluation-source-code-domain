import os
import sys
import random

atk_trainset_path = sys.argv[1]
res = {}

no = 8

orig_trainset = os.path.join(atk_trainset_path, 'data0.test.c2s')
total_lines = sum(1 for line in open(orig_trainset, 'r'))
for i in range(total_lines):
    res[i] = []

for i in range(1, no+1):
    with open('pred_wrong_lines' + str(i) + '.txt', 'r') as f:
        for line in f:
            lineno = int(line.strip()) #0-based
            res[lineno].append(i)

tf_trainset = [os.path.join(atk_trainset_path, 'data{}.test.c2s'.format(i)) for i in range(0,no+1)]
tf_trainset_fp = [open(filepath, 'r').readlines() for filepath in tf_trainset]

cnt = 0
univ_set = set(range(1, no+1))
with open('data1.train.c2s', 'w') as f:
    for i in range(total_lines):
        line_to_tf = random.randint(1,no)
        print(i, line_to_tf, tf_trainset[line_to_tf])
        f.write(tf_trainset_fp[line_to_tf][i])
