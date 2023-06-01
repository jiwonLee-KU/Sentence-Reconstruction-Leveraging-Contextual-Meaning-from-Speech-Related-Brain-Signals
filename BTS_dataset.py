import numpy as np
import hdf5storage
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import BTS_utils

def input_Neural_data(args):
    dir = args.data_root
    file_name = args.data_file_name
    data = hdf5storage.loadmat(dir + file_name)
    selected_task = 'os' if 'os' in file_name else 'is' if 'is' in file_name else None
    if selected_task:
        task_name = f'semantic_{selected_task}_EEG'
        task_y_name = f'{selected_task}_label'
        test_name = f'semantic_WIS_{selected_task}_EEG'
        test_y_name = f'WIS_{selected_task}_label'
        sentence_name = f'sentence_demo_{selected_task}_EEG'
        sentence_y_name = f'sentence_{selected_task}_label'
    x_data = data[task_name]
    x_test_data = data[test_name]
    x_sentence_data = data[sentence_name]
    # do common average reference (time by channel by trials)
    if args.CAR:
        print('CAR')
        x_data = x_data - np.mean(x_data, axis=1, keepdims=True)
    else:
        print('No CAR')
    # do z-score normalization at time (time by channel by trials)
    if args.zscore:
        print('z-score')
        x_data = x_data - np.mean(x_data, axis=0, keepdims=True)
        x_data = x_data / np.std(x_data, axis=0, keepdims=True)
    else:
        print('No z-score')
    y_data = data[task_y_name]
    y_test_data = data[test_y_name]
    y_sentence_data = data[sentence_y_name]
    x_sentence_data = np.transpose(x_sentence_data, [2, 1, 0])
    y_origin = y_data

    x_data = np.transpose(x_data, [2, 1, 0])
    for i in range(x_test_data.shape[0]):
        for j in range(x_test_data.shape[1]):
            x_test_data[i, j] = np.transpose(x_test_data[i, j], [2, 1, 0])

    return x_data, y_data, y_origin, x_test_data, y_test_data, x_sentence_data, y_sentence_data


def input_Neural_vector(args):
    dir = args.data_root
    file_name = args.data_file_name
    data = hdf5storage.loadmat(dir + file_name)
    selected_task = 'os' if 'os' in file_name else 'is' if 'is' in file_name else None
    if selected_task:
        task_name = f'semantic_{selected_task}_EEG'
        task_y_name = f'{selected_task}_label'
        test_name = f'semantic_WIS_{selected_task}_EEG'
        test_y_name = f'WIS_{selected_task}_label'
        sentence_name = f'sentence_demo_{selected_task}_EEG'
        sentence_y_name = f'sentence_{selected_task}_label'
    else:
        pass


    x_data = data[task_name]
    print("channel: ", x_data.shape[1])
    x_test_data = data[test_name]
    x_sentence_data = data[sentence_name]



    # common average reference (time by channel by trials)
    if args.CAR:
        print('CAR')
        x_data = x_data - np.mean(x_data, axis=1, keepdims=True)
        for i in range(x_test_data.shape[0]):
            for j in range(x_test_data.shape[1]):
                x_test_data[i,j] = x_test_data[i,j] - np.mean(x_test_data[i,j], axis=1, keepdims=True)
        # x_test_data = x_test_data - np.mean(x_test_data, axis=1, keepdims=True)
        x_sentence_data = x_sentence_data - np.mean(x_sentence_data, axis=1, keepdims=True)
    else:
        print('No CAR')
    # z-score normalization at time (time by channel by trials)
    if args.Zscore:
        print('z-score')
        x_data = x_data - np.mean(x_data, axis=0, keepdims=True)
        x_data = x_data / np.std(x_data, axis=0, keepdims=True)
        x_test_data = x_test_data - np.mean(x_test_data, axis=0, keepdims=True)
        x_test_data = x_test_data / np.std(x_test_data, axis=0, keepdims=True)
        x_sentence_data = x_sentence_data - np.mean(x_sentence_data, axis=0, keepdims=True)
        x_sentence_data = x_sentence_data / np.std(x_sentence_data, axis=0, keepdims=True)

    else:
        print('No z-score')

    y_data = data[task_y_name]
    y_test_data = data[test_y_name]
    y_test_origin = data[test_y_name]
    y_sentence_data = data[sentence_y_name]


    x_sentence_data = np.transpose(x_sentence_data, [2, 1, 0])

    y_origin = y_data
    y_origin = y_origin.squeeze()
    if args.word_embedding:
        y_test_data_label = [[[] for _ in range(y_test_data.shape[1])] for _ in range(y_test_data.shape[0])]
        y_test_data_vec = y_test_data_label
    else:
        print('No word2vec embedding')
        y_test_data_label = [[[] for _ in range(y_test_data.shape[1])] for _ in range(y_test_data.shape[0])]
        y_test_data_vec = y_test_data_label
    print("y_data_label_shape", y_data.shape)
    x_data = np.transpose(x_data, [2, 1, 0])
    for i in range(x_test_data.shape[0]):
        for j in range(x_test_data.shape[1]):
            if len(x_test_data[i, j].shape) == 3:
                x_test_data[i, j] = np.transpose(x_test_data[i, j], [2, 1, 0])
            elif len(x_test_data[i, j].shape) == 2:
                x_test_data[i, j] = np.transpose(x_test_data[i, j], [1, 0])
            else:
                print("Error: Data is not 2D or 3D!")
    args.n_class = len(np.unique(y_origin))
    print('unique: ', len(np.unique(y_origin)))

    if args.twodimconv:
        x_data = x_data[:, np.newaxis, :, :]
        for i in range(x_test_data.shape[0]):
            for j in range(x_test_data.shape[1]):
                x_test_data[i, j] = x_test_data[i, j][:, np.newaxis, :, :]
        x_sentence_data = x_sentence_data[:, np.newaxis, :, :]
        args.n_ch = x_data.shape[2]
        args.n_time = x_data.shape[3]
    else:
        args.n_ch = x_data.shape[1]
        args.n_time = x_data.shape[2]

    args.num_classes = len(np.unique(y_origin))
    args.input_channel = x_data.shape[1]
    args.input_time = x_data.shape[2]

    return x_data, y_data, y_origin, x_test_data, y_test_data_vec, y_test_origin, x_sentence_data, y_sentence_data



def data_augmentation(data, label, window_size=1000, shift_length=200):
    num_trials, num_channels, num_time_points = data.shape
    augmented_data = []
    augmented_label = []

    for trial in range(num_trials):
        for start in range(0, num_time_points - window_size + 1, shift_length):
            end = start + window_size
            windowed_data = data[trial, :, start:end]
            augmented_data.append(windowed_data)
            augmented_label.append(label[trial])

    augmented_data = torch.stack(augmented_data)
    augmented_label = torch.stack(augmented_label)
    return augmented_data, augmented_label


def sentence_windowing(sentence_data, label, word_size, shift_length=200):
    num_trials, num_channels, num_time_points = sentence_data.shape
    augmented_data = []
    augmented_label = []

    for trial in range(num_trials):
        for start in range(0, num_time_points - word_size + 1, shift_length):
            end = start + word_size
            windowed_data = sentence_data[trial, :, start:end]
            augmented_data.append(windowed_data)
            augmented_label.append(label[trial])

    augmented_data = np.stack(augmented_data)
    augmented_label = np.stack(augmented_label)
    return augmented_data, augmented_label



def windowing_training_data(x_data,y_data,args_window_size,args_overlap):
    # squeeze x_data to remove the channel dimension
    x_data = np.squeeze(x_data, axis=1)
    word_length = x_data.shape[2]
    print(word_length)
    print(args_window_size)
    print(args_overlap)
    windowing_x_data = np.zeros((1, x_data.shape[1], args_window_size))
    windowing_y_data = np.array([])
    for trials in range(0, x_data.shape[0]):
        print(trials)
        for i in range(0, int((word_length / args_overlap) - ((args_window_size - args_overlap) / args_overlap))):
            one_data = x_data[trials, :, i * args_overlap:i * args_overlap + args_window_size]
            one_data = one_data[np.newaxis, :, :]
            windowing_x_data = np.concatenate((windowing_x_data, one_data), axis=0)
            del one_data
            windowing_y_data = np.append(windowing_y_data, y_data[trials, 0])
    windowing_x_data = np.delete(windowing_x_data, 0, axis=0)
    windowing_x_data = windowing_x_data[:, np.newaxis, :, :]
    windowing_y_data = windowing_y_data[:, np.newaxis]
    print(windowing_x_data.shape)
    return windowing_x_data, windowing_y_data



def get_data(args):
    print("get_data")

def k_fold_dataset(x_data, y_origin, train_index, validation_index, fold, args, device, val_list):
    train_indices, test_indices = train_test_split(train_index, test_size=0.1,
                                                   stratify=y_origin[train_index],
                                                   random_state=0)
    setattr(val_list, 'train_indices_list', train_indices)
    setattr(val_list, 'val_indices_list', validation_index)
    setattr(val_list, 'test_indices_list', test_indices)
    X_train, X_val = x_data[getattr(val_list, "train_indices_list")[fold]], x_data[
        getattr(val_list, "val_indices_list")[fold]]
    Y_train, Y_val = y_origin[getattr(val_list, "train_indices_list")[fold]], y_origin[
        getattr(val_list, "val_indices_list")[fold]]
    X_test, Y_test = x_data[getattr(val_list, "test_indices_list")[fold]], y_origin[
        getattr(val_list, "test_indices_list")[fold]]
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_val = torch.tensor(Y_val, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32, device=device)
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                               pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def WIS_NS_dataset(x_data, y_origin,x_test_data,y_test_origin, sub_best_test_index, sub_tr_index, fold, args, device):
    X_train, X_val = x_data[sub_tr_index], x_data[sub_best_test_index]
    Y_train, Y_val = y_origin[sub_tr_index], y_origin[sub_best_test_index]
    tr_WIS_indices = [0, 1, 2, 3, 4]
    tr_WIS_indices.remove((fold % 5))
    tr_WIS_indices.remove(((fold + 1) % 5))

    if args.included_intestset:
        X_train = torch.tensor(np.concatenate((X_train, np.concatenate(
            [x_test_data[i][j] for j in tr_WIS_indices for i in range(x_test_data.shape[0])], axis=0)),
                                              axis=0), dtype=torch.float32, device=device)
        Y_train = torch.tensor(np.concatenate((Y_train, np.concatenate(
            [y_test_origin[i][j].squeeze() for j in tr_WIS_indices for i in
             range(x_test_data.shape[0])],
            axis=0)), axis=0), dtype=torch.float32, device=device)
        X_val = torch.tensor(np.concatenate((X_val, np.concatenate(
            [x_test_data[i][(fold % 5)] for i in range(x_test_data.shape[0])], axis=0)), axis=0),
                             dtype=torch.float32, device=device)
        Y_val = torch.tensor(np.concatenate((Y_val, np.concatenate(
            [y_test_origin[i][(fold % 5)].squeeze() for i in range(x_test_data.shape[0])], axis=0)),
                                            axis=0), dtype=torch.float32, device=device)
        X_test = torch.tensor(
            np.concatenate([x_test_data[i][((fold + 1) % 5)] for i in range(x_test_data.shape[0])],
                           axis=0),
            dtype=torch.float32, device=device)
        Y_test = torch.tensor(np.concatenate(
            [y_test_origin[i][((fold + 1) % 5)].squeeze() for i in range(x_test_data.shape[0])],
            axis=0),
            dtype=torch.float32, device=device)

    else:
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        Y_val = torch.tensor(Y_val, dtype=torch.float32, device=device)
        x_test_data_testset = np.concatenate(
            [x_test_data[i][j] for j in range(x_test_data.shape[0]) for i in
             range(x_test_data.shape[0])],
            axis=0)
        X_test = torch.tensor(x_test_data_testset, dtype=torch.float32, device=device)
        Y_test = torch.tensor(x_test_data_testset, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(X_train, Y_train)
    validation_set = TensorDataset(X_val, Y_val)
    test_set = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=False)
    valid_loader = DataLoader(validation_set, batch_size=args.batch_size,
                              shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
