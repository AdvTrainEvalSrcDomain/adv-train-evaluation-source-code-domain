import os
import sys
import glob
import random
from collections import defaultdict

inputdir = sys.argv[1]
dic1 = defaultdict(set)
dic2 = {}

for prog_path in glob.glob(os.path.join(inputdir, '**', '*'), recursive=False):
  if os.path.isdir(prog_path): continue
  basename = os.path.basename(prog_path)
  dic1['_'.join(basename.split('_')[:4])].add(prog_path)

for k, v in dic1.items():
  #print('v',v)
  ch = set([random.choice(list(v))])
  #print('ch',ch)
  diff = v - ch
  #print('a',diff)
  #print(diff if len(diff) != 0 else v)
  dic2[k] = diff

print(dic2)
for k, v in dic2.items():
  for item in v: os.remove(item)