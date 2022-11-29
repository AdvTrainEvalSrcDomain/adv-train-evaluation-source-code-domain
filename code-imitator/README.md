# Authorship Attribution
The code is modified based on [https://github.com/EQuiw/code-imitator](https://github.com/EQuiw/code-imitator) by Quiring et al. Please refer to the README of the original project (`src/PyProject/README.md`) for reference on environment preparation and data preprocessing.

## Training
### Normal model:
```
$ ./code-imitator/src/PyProject/evaluations/learning/rnn_css/start_train_models_rnn_parallel.sh
```

### Mixing Directly:
1. prepare the training set by adding adversarial examples into the original training set
2. modify `code-imitator/src/PyProject/evaluations/learning/rnn_css/vanilla_train.py` and `code-imitator/src/PyProject/evaluations/learning/rnn_css/MySequence.py` according to the `FIXME` instructions in the scripts (e.g., point the training set to the one in step 1)
3. run the script

### Composite Loss:
1. prepare the training set by adding adversarial examples into the original training set (their loss will be computed separately in the training)
2. modify `code-imitator/src/PyProject/evaluations/learning/rnn_css/train_comploss.py` according to the `FIXME` instructions in the script (e.g., point the training set to the one in step 1)
3. run the script

### Min-max + Composite Loss:
1. prepare the training set by adding adversarial examples AND perturbed examples into the original training set (their loss will be computed separately in the training)
2. modify `code-imitator/src/PyProject/evaluations/learning/rnn_css/train_minmax_comploss.py` according to the `FIXME` instructions in the script (e.g., change the max number of training epochs, after which new adversarial examples should be generated)
3. run the script, this will produce an intermediate model which we must attack to regenerate adversarial examples
4. attack the intermediate model and prepare new training set with the newly generated examples
5. modify `code-imitator/src/PyProject/evaluations/learning/rnn_css/train_minmax_comploss_cont.py` according to the `FIXME` instructions in the script
6. run the script with `train_minmax_comploss_cont.py <last trained keras model> <last trained model>`
7. repeat steps 4-6 until the desired number of times of example regeneration has been reached

### Min-max:
Same as Min-max + Composite Loss, except the scripts `code-imitator/src/PyProject/evaluations/learning/rnn_css/train_minmax.py` and `code-imitator/src/PyProject/evaluations/learning/rnn_css/train_minmax_cont.py` should be used instead

### Adversarial Fine-tuning:
1. prepare the training set by put all adversarial examples AND perturbed examples together (NO original clean examples) in one directory under respective author's subdirectores, call this directory `<adv-finetune-train>`
2. run `python random_pick.py <adv-finetune-train>` to ensure one adversarial example for each clean example
3. modify `code-imitator/src/PyProject/evaluations/learning/rnn_css/vanilla_train.py` according to the `FIXME` instructions in the script (e.g., point the training set to the one in step 1)
4. modify `code-imitator/src/PyProject/evaluations/learning/rnn_css/MySequence.py` according to `FIXME #11` in the script by changing the number to the number of epochs to fine-tune.
5. run `python vanilla_train.py trainnew <pre-trained keras model> <pre-trained model>`

## Adversarial Attack
To attack a model (for example, to attack the normal model and obtain adversarial examples for DA-like methods), place it at `code-imitator/data/ClassificationModels/CCS18_RNN/keras_model_<round ID>_<challenge ID>_RNN_800.pck` and then use `code-imitator/src/PyProject/evaluations/blackbox/attack/blackbox_attack_batch.py` and pass the problem ID and batch ID (1-17 for untargeted attacks and 1-19 for targeted attacks) as arguments. In this script, there is a line that says `IMPERSONATION = True`, which when set to `False` will set the attack mode to untargeted and `True` will be targeted. When you use the generated adversarial examples to synthesize new training set, make sure to name untargeted attack examples as `<original filename without extension>_mcts_unt.cpp`, and targeted attack examples as `<original filename without extension>_mcts_t.cpp`. In training involving composite loss, we will use the filename to relate the adversarial version to its original clean version.

For adversarially trained models, you may have to run `code-imitator/src/PyProject/add_final_test_into_model.py <model file> <path to testing set>` first to add testing set information into the model.

For fine-tuned models and intermediate models in OO-based methods, you may have to further run `code-imitator/src/PyProject/copy_cvec_into_final_train.py <pre-trained model> <fine-tuned model>` to add the missing vectorizer into the model.

## Testing
### Effectiveness (Accuracy and F1-Score):
Modify `code-imitator/src/PyProject/evaluations/learning/rnn_css/test.py` according to the `FIXME` instructions in the script, and then run it with `test.py <model> <keras model>`.

### Robustness (ASRunt):
Put all directories generated by the MCTS attack (whose names are like `blackbox_3264486_5633382285312000_1_CCS18_RNN_800_False_MCTS_Classic`) under a specific directory. Run `code-imitator/src/PyProject/count_by_numof_tfs.py <directory path> 3`. The last line of output will be the ASRunt. In previous lines, some filenames, numbers, and `True/False` will be shown, which means for the specified file, the stated number of transformations was applied, and finally the attack on this file was successfully done within perturbation budget (`True`) or not (`False`). This information can help you distinguish adversarial examples and perturbed examples.

### Robustness (ASRtar):
Put all directories generated by the MCTS attack (whose names are like `blackbox_3264486_5633382285312000_1_CCS18_RNN_800_True_MCTS_Classic_True`) under a specific directory. Run `code-imitator/src/PyProject/t_count_by_numof_tfs.py <directory path> 3`. The last line of output will be the ASRtar. In previous lines, some filenames, numbers, and `True/False` will be shown. These have the same meaning as above.