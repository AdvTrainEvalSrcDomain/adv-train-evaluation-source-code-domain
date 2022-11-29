import os
import sys

atk_trainset_path = sys.argv[1]
res = {}

no = 8

for i in range(no, no+1):
    with open('pred_wrong_lines' + str(i) + '.txt', 'r') as f:
        for line in f:
            lineno = int(line.strip())
            if res.get(lineno, None) is None:
                res[lineno] = i

orig_trainset = os.path.join(atk_trainset_path, 'data0.test.c2s')
total_lines = sum(1 for line in open(orig_trainset, 'r'))
tf_trainset = [os.path.join(atk_trainset_path, 'data{}.test.c2s'.format(i)) for i in range(0,no+1)]
tf_trainset_fp = [open(filepath, 'r').readlines() for filepath in tf_trainset]

with open('data' + str(no) + '.train.c2s', 'w') as f:
    for i in range(total_lines):
        line_to_tf = res.get(i, 0)
        print(i, line_to_tf, tf_trainset[line_to_tf])
        f.write(tf_trainset_fp[line_to_tf][i])
