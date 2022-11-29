Raw java-small data has been put under `raw/c2s/java-small` (downloaded by running `sudo make download-datasets` under project root directory). But the raw data is unfolded (but train/test split has been done), so it will serve as the 1st fold. The remaining folds need to be created manually (the complete dataset including preprocessed data is huge, so we don't put it here):
1. download the original java-small dataset from [https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz](https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz) and unzip it (path to the unzipped files will be called `<original-data>`)
2. in `<original-data>`, there are 10 subdirectories in `training` and `test` in total. Move each of them into `test` once (and move what was in `test` to `training`), and this will be one fold.
3. when each of 10 subdirectories has served as `test` once, all 10 folds will have been created (actually 9 created because the 1st fold is done in step 1)
4. process the original data into raw form by running `python process_raw.py` with the first argument being the name of each directory (NOT subdirectory) under `<original-data>`
5. rename the resulting `result.jsonl` into `<name of directory>.jsonl`
6. run `gzip <name of directory>.jsonl`
7. put renamed jsonl files into `datasets/raw/c2s/java-small`
8. do the rest of preprocessing, training etc according to the main README (also see `README_orig.md` under this directory for a description of the forms of data used)
