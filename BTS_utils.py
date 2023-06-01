import torch
import numpy as np
import os
import pandas as pd
import scipy.io
from sklearn.metrics import confusion_matrix
import BTS_confusionmatrix




def load_best_val_acc_model(path):

    files = os.listdir(path)
    files = [f for f in files if f.endswith('.ckpt')]


    files.sort(key=lambda f: float(f.split('val_acc=')[1].split('.ckpt')[0]), reverse=True)
    val_best = files[0].split('val_acc=')[1].split('.ckpt')[0]
    val_acc_files = [f for f in files if val_best in f]

    max_epoch = -1
    max_epoch_file = None

    for f in val_acc_files:
        epoch_str = f.split('=')[1].split('_')[0]
        epoch = int(epoch_str)
        if epoch > max_epoch:
            max_epoch = epoch
            max_epoch_file = f

    if max_epoch_file is not None:
        print(f"'val_acc='가 가장 높은 파일 중에서 'epoch='가 가장 높은 파일: {max_epoch_file}")
    else:
        print("검색된 파일이 없습니다.")

    file_path = f"{path}/{max_epoch_file}"


    return file_path




def split_sentence_WIS_true_labels(WIS_true_word, start_word):
    splitted_labels = []
    current_list = []
    for word in WIS_true_word:
        if word in start_word:
            if current_list:
                splitted_labels.append(current_list)
                current_list = []
            current_list.append(word)
        else:
            current_list.append(word)

    if current_list:
        splitted_labels.append(current_list)

    return splitted_labels


def split_sentence_WIS_pred_labels(WIS_pred_word, splited_sentence_WIS):
    splitted_preded_word = []
    i = 0
    for word_length in splited_sentence_WIS:
        n = len(word_length)
        splitted_preded_word.append(WIS_pred_word[i:i + n])
        i += n

    return splitted_preded_word




def save_results(folder_name, train_name, all_preds, all_trues, test_accs, num_folds, labels, args, test_preds_list,test_true_list, WIS_WER_list, WIS_WER_NLP_list, WIS_pred_list, WIS_corrected_pred_list, WIS_true_list,WIS_pred_sen_list,WIS_corrected_pred_sen_list,WIS_true_sen_list,WIS_SER,WIS_corrected_SER, sen_SER_list, true_sen_list, pred_sen_list, pred_word_sen_list, pred_similarity_sen_list):
    np.save(f"{folder_name}/{train_name}_pred.npy", all_preds)
    np.save(f"{folder_name}/{train_name}_true.npy", all_trues)
    WIS_pred_sen_list = [WIS_pred_sen_list[i:i + 5] for i in range(0, len(WIS_pred_sen_list), 5)]
    WIS_corrected_pred_sen_list = [WIS_corrected_pred_sen_list[i:i + 5] for i in range(0, len(WIS_corrected_pred_sen_list), 5)]
    WIS_true_sen_list = [WIS_true_sen_list[i:i + 5] for i in range(0, len(WIS_true_sen_list), 5)]

    if args.word_embedding:
        acc_cosinesim = compute_accuracies(test_true_list, test_preds_list)
        df = pd.DataFrame({'fold': range(num_folds), 'test_acc': test_accs, 'test_acc_cosinesim': acc_cosinesim,
                           'WER': WIS_WER_list, 'WER_NLP': WIS_WER_NLP_list, 'WIS_true': WIS_true_list, 'WIS_pred': WIS_pred_list, 'WIS_corrected_pred': WIS_corrected_pred_list,'WIS_pred_sen_list' : WIS_pred_sen_list, 'WIS_corrected_pred_sen_list': WIS_corrected_pred_sen_list,'WIS_true_sen_list':WIS_true_sen_list,'WIS_SER': WIS_SER, 'WIS_corrected_SER':WIS_corrected_SER, 'SER': sen_SER_list, 'true_sen': true_sen_list, 'pred_sen': pred_sen_list, 'pred_word_sen': pred_word_sen_list, 'pred_similarity_sen': pred_similarity_sen_list})
    else:
        df = pd.DataFrame({'fold': range(num_folds), 'test_acc': test_accs,'WER': WIS_WER_list, 'WER_NLP': WIS_WER_NLP_list, 'WIS_true': WIS_true_list, 'WIS_pred': WIS_pred_list, 'WIS_corrected_pred': WIS_corrected_pred_list,'WIS_pred_sen_list' : WIS_pred_sen_list, 'WIS_corrected_pred_sen_list': WIS_corrected_pred_sen_list,'WIS_true_sen_list':WIS_true_sen_list, 'WIS_SER': WIS_SER, 'WIS_corrected_SER':WIS_corrected_SER, 'SER': sen_SER_list, 'true_sen': true_sen_list, 'pred_sen': pred_sen_list, 'pred_word_sen': pred_word_sen_list, 'pred_similarity_sen': pred_similarity_sen_list})

    df.to_excel(f"{folder_name}/demo_{train_name}.xlsx", index=False)


def save_results_k_fold(folder_name, train_name, all_preds, all_trues, test_accs, num_folds, labels, args, test_preds_list,test_true_list):
    np.save(f"{folder_name}/{train_name}_pred.npy", all_preds)
    np.save(f"{folder_name}/{train_name}_true.npy", all_trues)

    if args.word_embedding:
        acc_cosinesim = compute_accuracies(test_true_list, test_preds_list)
        df = pd.DataFrame({'fold': range(num_folds), 'test_acc': test_accs, 'test_acc_cosinesim': acc_cosinesim})
    else:
        df = pd.DataFrame({'fold': range(num_folds), 'test_acc': test_accs})

    df.to_excel(f"{folder_name}/demo_{train_name}.xlsx", index=False)


def compute_accuracies(test_true_list, test_preds_list):
    all_true = np.array(test_true_list)
    all_pred = np.array(test_preds_list)
    acc_cosinesim = []

    for i in range(len(all_true)):
        acc_cosinesim.append(sum(all_true[i] == all_pred[i]) / len(all_true[i]) * 100)

    return np.array(acc_cosinesim)


def process_confusion_matrices(num_folds, folder_name, train_name, all_trues, all_preds, labels, args):
    result_confusion_matrix = confusion_matrix(all_trues, all_preds)
    BTS_confusionmatrix.plot_confusion_matrix(result_confusion_matrix, labels,
                                              f"{folder_name}/{train_name}_result_confusion_matrix.png")
    scipy.io.savemat(f"{folder_name}/{train_name}_confusion_matrix.mat",
                     mdict={'result_confusion_matrix': result_confusion_matrix})
    scipy.io.savemat(f"{folder_name}/{train_name}_pred_label.mat", mdict={'all_pred_label': all_preds})
    scipy.io.savemat(f"{folder_name}/{train_name}_true_label.mat", mdict={'all_true_label': all_trues})

def process_cosinesimilarties_matrices(num_folds, folder_name, train_name, all_trues, all_preds, labels, args):
    result_confusion_matrix = confusion_matrix(all_trues, all_preds)
    BTS_confusionmatrix.plot_confusion_matrix(result_confusion_matrix, labels,
                                              f"{folder_name}/{train_name}_result_confusion_matrix.png")
    scipy.io.savemat(f"{folder_name}/{train_name}_confusion_matrix.mat",
                     mdict={'result_confusion_matrix': result_confusion_matrix})
    scipy.io.savemat(f"{folder_name}/{train_name}_pred_label.mat", mdict={'all_pred_label': all_preds})
    scipy.io.savemat(f"{folder_name}/{train_name}_true_label.mat", mdict={'all_true_label': all_trues})


def find_matching_sentence(str_variable, sentences):
    words = [sentence.lower().split() for sentence in sentences]

    word_count = {}
    for sentence in str_variable.lower().split('.'):
        for word in sentence.split():
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1

    sentence_count = [0] * len(sentences)
    for i in range(len(sentences)):
        for word in words[i]:
            if word in word_count:
                sentence_count[i] += word_count[word]

    max_count = max(sentence_count)
    if max_count == 0:
        return "No matching sentence found."
    else:
        max_index = sentence_count.index(max_count)
        return sentences[max_index]

def load_best_index(args, sub):
    sub_best_fold_index = np.load(
        f"{args.result_index_root}/sub{args.sub_list[sub - 1]}_{args.task_selected}_fold_good_index_one_hot.npz")
    sub_best_val_index = sub_best_fold_index['best_val_indices']
    sub_best_tr_index = sub_best_fold_index['best_train_indices']
    sub_best_test_index = sub_best_fold_index['best_test_indices']
    sub_tr_index = np.concatenate((sub_best_val_index, sub_best_tr_index))
    return sub_best_test_index, sub_tr_index

def make_save_folder(args, path):
    folder_name = path
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        i = 2
        while True:
            new_folder_name = f"{folder_name}_new_ver{i}"
            if not os.path.exists(new_folder_name):
                os.makedirs(new_folder_name)
                folder_name = new_folder_name
                print(f"A new folder has been created: {new_folder_name}")
                break
            i += 1
    if args.word_embedding:
        similarity_folder = f"{folder_name}/cosine_similarity"
    else:
        similarity_folder = f"{folder_name}/softmax"
    if not os.path.exists(similarity_folder):
        os.makedirs(similarity_folder)

    return folder_name, similarity_folder


class Val_list:
    def __init__(self):
        self.test_accs = []
        self.test_preds_list = []
        self.test_preds_e_list = []
        self.test_true_list = []
        self.WIS_WER_list = []
        self.WIS_pred_list = []
        self.WIS_true_list = []
        self.WIS_corrected_pred_list = []
        self.WIS_WER_NLP_list = []

        self.WIS_true_sen_list = []
        self.WIS_pred_sen_list = []
        self.WIS_corrected_pred_sen_list = []
        self.fold_sen_SER_list = []
        self.sen_SER_list = []
        self.fold_true_sen_list = []
        self.fold_pred_sen_list = []
        self.fold_pred_word_sen_list = []
        self.fold_similar_sen_list = []
        self.pred_step_word_index = []
        self.pred_step_sentence_index = []
        self.pred_step_word_similarity = []
        self.pred_word_sen_list = []
        self.pred_similarity_sen_list = []
        self.true_sen_list = []
        self.pred_sen_list = []
        self.train_indices_list = []
        self.val_indices_list = []
        self.test_indices_list = []


    def __setattr__(self, key, value):
        if hasattr(self, key):
            getattr(self, key).append(value)
        else:
            super().__setattr__(key, value)

    def reset(self):
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                value.clear()

    def add_to_list(self, key, value):
        if hasattr(self, key):
            getattr(self, key).append(value)
        else:
            super().__setattr__(key, [value])