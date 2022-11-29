import os
import sys
import glob
from distutils.dir_util import copy_tree
import shutil

dir_list = sys.argv[1]
for prog_path in glob.glob(os.path.join(dir_list, '*'), recursive=False):
  prog = os.path.basename(prog_path)
  if os.path.isdir(prog_path): continue
  src_author = prog.split('.cpp')[0].split('_')[2]
  src_author_dir = os.path.join(dir_list, src_author)
  if not os.path.exists(src_author_dir): os.makedirs(src_author_dir)
  cpy_dst = os.path.join(src_author_dir, prog)
  print(prog_path, cpy_dst)
  try:
    shutil.copy(prog_path, cpy_dst)
    os.remove(prog_path)
  except:
    pass
  # shutil.rmtree(subdir_path)
