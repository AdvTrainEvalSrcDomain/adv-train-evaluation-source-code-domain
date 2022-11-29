# Code Summarization
The code is modified based on [https://github.com/jjhenkel/averloc](https://github.com/jjhenkel/averloc) by Hanke et al. Please refer to the README of the original project (`README_orig.md`) for reference on environment preparation and data preprocessing.

## Training
### Normal model:
```
$ sudo ARGS="--regular_training --epochs 20" GPU=0 MODELS_OUT=final-models/code2seq/c2s/java-small DATASET_NAME=datasets/preprocessed/ast-paths/c2s/java-small time make train-model-code2seq
```

### Mixing Directly:
1. train a normal model
2. attack the normal model on the training set (see below "Adversarial Attack")
3. assuming the results of step 2 are in the directory `<atk-on-normal>`, create a subdirectory under it to store the transformed training set examples. This subdirectory will be referred to as `<atk-on-train>`. Further create a subdirectory `adv` under `<atk-on-train>` to store adversarial examples
4. move all `datax.train.c2s` (where x=0..8) under `<atk-on-normal>` to `<atk-on-train>`
5. copy `data.dict.c2s` and `data.val.c2s` into `<atk-on-train>` and `<atk-on-train>/adv`
6. now we evaluate these transformed examples to find out which ones are adversarial (attacked successfully). First rename all `datax.train.c2s` into `datax.test.c2s`
7. rename each of `datax.test.c2s` to `data0.test.c2s`, one at a time (i.e., first rename `data0.test.c2s`, which is already that name so do nothing, and carry out steps 8-10; then rename `data1.test.c2s` to `data0.test.c2s` (back up the original first), and do 8-10, etc)
8. run the following, where `<x>` is the number in the filename you renamed in step 7
```
$ sudo GPU=0 MODELS_IN=final-models/code2seq/c2s/java-small/normal ARGS="1" \
DATASET_NAME=<atk-on-train> \
RESULTS_OUT=final-results/code2seq/c2s/java-small/normal-model/depth-1-gradient-attack-on-tf<x> \
time make test-model-code2seq
```
9. run the following, where `<x>` is the number in the filename you renamed in step 7 (skip this and next step if x=0)
```
$ cd final-results/code2seq/c2s/java-small/normal-model
$ python acc.py 1 depth-1-gradient-attack-on-tf0 depth-1-gradient-attack-on-tf<x>
```
10. rename `sed_cmd.txt` into `sed_cmd<x>.txt` and `pred_wrong_lines.txt` into `pred_wrong_lines<x>.txt`, repeat steps 7-10 for each `x`
11. go back to project root directory and run `make_simple_augmented_trainset.sh final-results/code2seq/c2s/java-small/normal-model <atk-on-train>`
12. copy `<atk-on-train>/data0.test.c2s` to `<atk-on-train>/adv/data0.test.c2s` and run `cat data0.test.c2s data1.test.c2s data2.test.c2s data3.test.c2s data4.test.c2s data5.test.c2s data6.test.c2s data7.test.c2s data8.test.c2s > data.train.c2s`
13. replace `models/code2seq-merged/model.py` with `models/code2seq-merged/model_gradclip.py` to introduce gradient clipping (replace it back after Mixing Directly training is finished)
14. now the training set has been prepared, run the following to start training
```
$ sudo ARGS="--regular_training --epochs 20" GPU=0 MODELS_OUT=final-models/code2seq/c2s/java-small DATASET_NAME=<atk-on-train>/adv time make aug-train-model-code2seq
```

### Composite Loss:
1. finish steps 1-10 in Mixing Directly
2. run `python make_comp_loss_trainset.py <atk-on-train>`
3. move the generated `data1.train.c2s` to a new directory (create one) under `datasets/adversarial`, this will be referred to as `<comploss-train>`
4. copy `data.dict.c2s` and `data.val.c2s` under `<atk-on-normal>` to `<comploss-train>`
5. run the following to start training
```
$ sudo ARGS="2 --epochs 20 --lamb 0.4" GPU=0 MODELS_OUT=final-models/code2seq/c2s/java-small DATASET_NAME=<comploss-train> time make aug-train-model-code2seq
```

### Min-max + Composite Loss:
```
$ sudo GPU=0 DATASET=c2s/java-small ./scripts/per-epoch-adv-train-code2seq.sh --gradient
```

### Min-max: 
```
$ sudo GPU=0 DATASET=c2s/java-small ./scripts/per-epoch-adv-train-code2seq-l0.0.sh --gradient
```

### Adversaral Fine-tuning:
1. finish steps 1-10 in Mixing Directly
2. modify line 7 `no = 8` in `make_finetune_with_adv_examples_trainset.py` as such: change the number from 1 to 8, one at a time
3. for each value of `no`, run `python make_finetune_with_adv_examples_trainset.py <atk-on-train>`
4. when all values are finished, move the generated `datax.train.c2s` (x=1..8) to a new directory (create one) under `datasets/adversarial`, this will be referred to as `<adv-finetune-train>`
5. copy `data.dict.c2s` and `data.val.c2s` under `<atk-on-normal>` to `<adv-finetune-train>`
6. run the following to start training 
```
$ sudo ARGS="--adv_fine_tune2 9 --epochs 1 --lamb 0.0" GPU=0 MODELS_OUT=final-models/code2seq/c2s/java-small DATASET_NAME=<adv-finetune-train> time make adv-train-model-code2seq2
```

## Adversarial Attack
only on testing set:
```
$ sudo GPU="0" \
NO_RANDOM="true" \
NO_GRADIENT="false" \
NO_TEST="false" \
AVERLOC_JUST_TEST="true" \
SHORT_NAME="<path to store the transformed examples into>" \
DATASET="c2s/java-small" \
MODELS_IN="<path to the model to be attacked>" \
TRANSFORMS="transforms\.\w+" \
  time make extract-adv-dataset-ast-paths
```

on both training and testing sets:
```
$ sudo GPU="0" \
NO_RANDOM="true" \
NO_GRADIENT="false" \
NO_TEST="false" \
AVERLOC_JUST_TEST="false" \
SHORT_NAME="<path to store the transformed examples into>" \
DATASET="c2s/java-small" \
MODELS_IN="<path to the model to be attacked>" \
TRANSFORMS="transforms\.\w+" \
  time make extract-adv-dataset-ast-paths
```

`<path to store the transformed examples into>` is relative to `datasets/adversarial`

## Testing
### Effectiveness (Accuracy):
```
$ sudo GPU=0 MODELS_IN=<model to be tested> ARGS="1" \
DATASET_NAME=<test set path> \
RESULTS_OUT=final-results/code2seq/c2s/java-small/<give a name for this model>/no-attack \
time make test-model-code2seq
```

### Effectiveness (F1-Score):
```
$ sudo GPU=0 MODELS_IN=<model to be tested> ARGS="--no-attack" DATASET_NAME=datasets/preprocessed/ast-paths/c2s/java-small RESULTS_OUT=final-results/code2seq/c2s/java-small/<give a name for this model>/no-attack   time make test-model-code2seq
```

### Robustness (ASRunt):
```
$ sudo GPU=0 MODELS_IN=<model to be tested> ARGS="9" \
DATASET_NAME=<test set path> \
RESULTS_OUT=final-results/code2seq/c2s/java-small/<give a name for this model>/depth-1-gradient-attack \
time make test-model-code2seq
$ python models/pytorch-seq2seq/seq2seq/evaluator/metrics.py \
--f_true final-results/code2seq/c2s/java-small/<give a name for this model>/depth-1-gradient-attack/true_target \
--f_pred final-results/code2seq/c2s/java-small/<give a name for this model>/depth-1-gradient-attack/predicted_target \
--o_pred final-results/code2seq/c2s/java-small/<give a name for this model>/no-attack/predicted_target
```
