import os
import sys

def cmp_mode1(orig_true, orig_pred, atk_pred):
    orig_pred_correct_lineno = []
    orig_pred_correct = []
    atk_pred_incorrect_lineno = []
    atk_pred_incorrect = []
    total_lineno = 0
    orig_pred_correct_subtokens = 0
    with open(orig_true, 'r') as f, open(orig_pred, 'r') as g:
        for idx, (linef, lineg) in enumerate(zip(f, g)):
            total_lineno += 1
            orig_pred_result = lineg
            if linef.strip() == orig_pred_result.strip():
                orig_pred_correct_lineno.append(idx); orig_pred_correct.append(linef.strip())
            linef_split = linef.split(); lineg_split = lineg.split()
            for i in range(min(len(linef_split), len(lineg_split))):
                if linef_split[i] == lineg_split[i]: orig_pred_correct_subtokens += 1
    print('orig acc: ', len(orig_pred_correct_lineno) / total_lineno)

    atk_pred_incorrect_subtokens = 0
    with open(atk_pred, 'r') as f, open(orig_true, 'r') as g, open(orig_pred, 'r') as h:
        for idx, (line, lineg, lineh) in enumerate(zip(f, g, h)):
            # if idx in orig_pred_correct_lineno:
            #     atk_pred_result = line.strip()
            #     orig_pred_result = orig_pred_correct[orig_pred_correct_lineno.index(idx)]
            #     if atk_pred_result != orig_pred_result:  atk_pred_incorrect_lineno.append(idx); atk_pred_incorrect.append(atk_pred_result)
            #     orig_pred_result_split = orig_pred_result.split(); atk_pred_result_split = atk_pred_result.split()
            #     for i in range(min(len(orig_pred_result_split), len(atk_pred_result_split))):
            #         if orig_pred_result_split[i] != atk_pred_result_split[i]: atk_pred_incorrect_subtokens += 1
            #     #if len(atk_pred_result_split) > len(orig_pred_result_split): atk_pred_incorrect_subtokens += (len(atk_pred_result_split) - len(orig_pred_result_split))
            # else:
            atk_pred_result = line.strip()
            orig_true_result = lineg.strip()
            orig_pred_result = lineh.strip()
            if orig_true_result == orig_pred_result and orig_true_result != atk_pred_result: atk_pred_incorrect_lineno.append(idx); atk_pred_incorrect.append(atk_pred_result)
            orig_true_result_split = orig_true_result.split(); atk_pred_result_split = atk_pred_result.split(); orig_pred_result_split = orig_pred_result.split()
            for i in range(min(len(orig_true_result_split), len(atk_pred_result_split), len(orig_pred_result_split))):
                if orig_true_result_split[i] == orig_pred_result_split[i] and orig_true_result_split[i] != atk_pred_result_split[i]: atk_pred_incorrect_subtokens += 1
            #if len(atk_pred_result_split) > len(orig_true_result_split): atk_pred_incorrect_subtokens += (len(atk_pred_result_split) - len(orig_true_result_split))
    sed_cmd = ""
    pred_wrong_lines = ""
    for i in range(len(atk_pred_incorrect_lineno)):
        idx = atk_pred_incorrect_lineno[i]
        print('atk_wrong_lineno: {}, orig_pred: {}, atk_pred: {}'.format(idx, orig_pred_correct[orig_pred_correct_lineno.index(idx)], atk_pred_incorrect[i]))
        sed_cmd += str(idx+1) + "p;"
        pred_wrong_lines += str(idx) + '\n'

    with open('sed_cmd.txt', 'w') as f: f.write(sed_cmd)
    with open('pred_wrong_lines.txt', 'w') as f: f.write(pred_wrong_lines)

    print('orig acc: ', len(orig_pred_correct_lineno) / total_lineno)
    print('untargeted asr: ', len(atk_pred_incorrect_lineno) / len(orig_pred_correct_lineno))
    print('untargeted asr (subtoken-level): ', atk_pred_incorrect_subtokens / orig_pred_correct_subtokens)

mode = sys.argv[1]
orig_dir = sys.argv[2]
atk_dir = sys.argv[3]

if mode == '1':
    #compare two pairs of true_target and predicted_target
    orig_true = os.path.join(orig_dir, 'true_target')
    orig_pred = os.path.join(orig_dir, 'predicted_target')
    atk_pred = os.path.join(atk_dir, 'predicted_target')
    cmp_mode1(orig_true, orig_pred, atk_pred)
elif mode == '2':
    #compare a normal testing result with a pair of true_target and predicted_target
    pass