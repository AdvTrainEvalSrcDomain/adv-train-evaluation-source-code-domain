# Introduction
This repository is dedicated to our paper "A Comparative Study of Adversarial Training Methods for Neural Models of Code". We evaluate five adversarial training methods on two tasks (code summarization and authorship attribution) using one model for each task (code2seq and Abuhamad). 

Adversarial training has been employed by researchers to protect AI models of source code. However, it is still unknown how adversarial training methods in this field compare to each other in effectiveness and robustness. This study surveys and investigates existing adversarial training methods, and conducts experiments to evaluate these neural models' performance in the domain of source code. First, we examine the process of adversarial training to identify four dimensions that could be used to classify different adversarial training methods into five categories, which are Mixing Directly, Composite Loss, Adversarial Fine-tuning, Min-max + Composite Loss, and Min-max. Second, we conduct empirical evaluations of these classified adversarial training methods under two tasks (i.e., code summarization and code authorship attribution) to determine their performance of effectiveness and robustness. Experimental results indicate that the performance of certain combinations of adversarial training techniques (i.e., min-max with composite loss, or directly-sample with ordinary loss) would be much better than other combinations or other techniques used alone. Our experiment also reveals that the model's robustness of defensive methods can be enhanced by using diverse input data for adversarial training, and that fine-tuning epochs has little or no impact on model's performance. 

# Project Structure
The files for code summarization are under the directory `averloc` and files for authorship attribtion are under `code-imitator`. Some of the more important files in this project are listed below.
```
averloc                                       # files for code summarization
  -- datasets                                 # stores datasets for code summarization
  -- models                                   # code of code2seq model
  -- scripts                                  # auxiliary scripts, including script for Min-max (+ Comp. Loss) training
  -- tasks                                    # Docker scripts
  -- acc.py                                   # auxiliary script for Mixing Directly
  -- make_comp_loss_trainset.py               # script for making training set for Composite Loss
  -- make_comp_loss_with_advperb_trainset.py  # script for making training set for Composite Loss + pert. examples
  -- make_finetune_with_adv_trainset.py       # script for making training set for Adversarial Fine-tuning
  -- make_simple_augmented_trainset.sh        # script for making training set for Mixing Directly
  -- Makefile                                 # defines various make commands, including commands for training and testing

code-imitator                                 # files for authorship attribution
  -- data                                     # stores datasets for authorship attribution
  -- src
    -- PyProject                              # where the main code is
      -- evaluations
        -- learning
          -- rnn_css
            -- vanilla_train.py               # script for Mixing Directly and Adversarial Fine-tuning
            -- train_comploss.py              # script for Composite Loss
            -- train_minmax.py                # script for Min-max
            -- train_minmax_cont.py           # script for Min-max
            -- train_minmax_comploss.py       # script for Min-max + Comp. Loss
            -- train_minmax_comploss_cont.py  # script for Min-max + Comp. Loss
            -- test.py                        # script for evaluating accuracy and F1-score
```

# Usage
The details on how to evaluate adversarial training methods are described in the README files in the directories of respective tasks. For guidance on how to set up environment before experiments can be carried out, please refer to the README of the original projects.

