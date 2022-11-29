import os
import sys
import glob

input_dir = sys.argv[1]

nosuccess_list = []
within_budget_list = []
sum = 0
max_tfs = int(sys.argv[2], 10)
all_log_path = glob.glob(input_dir + "/*/*/*/debug.log")
all_log_path_len = len(all_log_path)
for log_path in all_log_path:
  with open(log_path, 'r') as f:
    found = False
    include_error = False
    success = False
    nosuccess = False
    exp_len = 0
    for line in f:
      if 'NOSUCCESS' in line: nosuccess = True
      elif 'SUCCESS' in line: success = True
      if 'AddTemplateTransformer includeadd_transformer has produced error output' in line or 'Break Up due to empty children' in line:
        include_error = True
        break
      if found:
        if exp_len == 0:
          exp_len = line.count(',')-1
        else:
          if 'Selection node depth' in line:
            sel_depth = int(line.split(': ')[-1], 10)
      if 'Expansion Sequence' in line: found = True
    if include_error or not (success or nosuccess):
      all_log_path_len -= 1
      continue
    if found:
      print(log_path, exp_len+sel_depth, exp_len+sel_depth <= max_tfs and success)
      if exp_len+sel_depth <= max_tfs: within_budget_list.append(log_path.split('/')[-3])
      if exp_len+sel_depth <= max_tfs and success: sum += 1
      else: nosuccess_list.append(log_path.split('/')[-3])
    else:
      f.seek(0, 0)
      for line in f:
        if 'Next outer node id:' in line:
          depth = int(line.split('depth: ')[-1], 10)
      try:
        print(log_path, depth, depth <= max_tfs and success)
        if depth <= max_tfs: within_budget_list.append(log_path.split('/')[-3])
        if depth <= max_tfs and success: sum += 1
        else: nosuccess_list.append(log_path.split('/')[-3])
      except NameError:
        pass

print(sum)
print(all_log_path_len)
print(sum / all_log_path_len)
nosuccess_list.sort()
with open('ll.txt', 'w') as f:
  for item in nosuccess_list: f.write(os.path.join(input_dir, 'mcts/aaa/3264486_5633382285312000_' + item + '_mcts_unt_seed40.cpp') + '\n')
with open('test.txt', 'w') as f:
    for item in within_budget_list: f.write('3264486_5633382285312000_' + item + '_mcts_unt_seed40.cpp\n')