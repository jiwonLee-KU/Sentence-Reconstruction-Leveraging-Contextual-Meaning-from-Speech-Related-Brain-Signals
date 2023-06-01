import BTS_dataset
import BTS_utils
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from absl import flags
import BTS_nnembedding_generator
import random
import scipy.io
import BTS_confusionmatrix
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import BTS_LM

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, predicted, target):
        similarity = self.cosine_similarity(predicted, target)
        loss = 1 - similarity.mean()
        return loss


class HuberLoss:
    def __init__(self, delta=1.0):
        self.delta = delta
    def __call__(self, y_pred, y_true):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        is_small_error = abs_error <= self.delta
        squared_loss = 0.5 * torch.square(error)
        linear_loss = self.delta * (abs_error - 0.5 * self.delta)
        return torch.where(is_small_error, squared_loss, linear_loss).mean()


class seq2seq_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        # global vec_dimension
        self.num_classes = args.num_classes
        self.input_channel = args.input_channel
        self.input_time = args.input_time
        # self.ndim = vec_dimension
        self.ndim = args.vec_dim
        print('vector dimension is: ', self.ndim)
        # self.generator = BTS_nnembedding_generator.Generator(self.input_channel, self.ndim,
        #                                                           num_channels=[self.input_channel, self.input_channel,
        #                                                                         self.input_channel, self.input_channel,
        #                                                                         self.input_channel, self.input_channel, self.input_channel,
        #                                                                         256, 256], kernel_size=9, dropout=0.0)
        self.generator = BTS_nnembedding_generator.Generator(self.input_channel, self.ndim,
                                                            num_channels=[self.input_channel, self.input_channel,
                                                                        self.input_channel,
                                                                        256, 256], kernel_size=7, dropout=0.0)
        self.discriminator = BTS_nnembedding_generator.Vec2onehot_Discriminator()
        if args.word_embedding:
            self.MSE_criteria = nn.MSELoss()
            self.criteria = nn.CrossEntropyLoss()
        else:
            self.criteria = nn.CrossEntropyLoss()

    def forward(self, z):
        return self.generator(z)

    def ved2onehot_loss(self, y_hat, y):
        return nn.CrossEntropyLoss(y_hat, y)


    def loss(self, y_hat, y):
        return self.criteria(y_hat, y)


    def accuracy(self, y_hat, y):
        similarities = F.cosine_similarity(y_hat.unsqueeze(1), embedding_labels_model.unsqueeze(0), dim=2)
        true_similarities = F.cosine_similarity(y.unsqueeze(1), embedding_labels_model.unsqueeze(0), dim=2)
        most_similar_index = similarities.argmax(dim=1)
        true_similar_index = true_similarities.argmax(dim=1)
        correct_predictions = (most_similar_index == true_similar_index).sum().item()
        total_predictions = y.size(0)
        accuracy = correct_predictions / total_predictions
        converted_true_label = []
        converted_true_label.append(true_similar_index)
        true = converted_true_label
        cosinesim_pred_label = []
        cosinesim_pred_label.append(most_similar_index)
        preds = cosinesim_pred_label
        return accuracy, true, preds

    def predicted_word_labels(self, y_hat):
        global cosine_similarity_distance
        cosine_similarity_distance = []
        similarities_word = F.cosine_similarity(y_hat.unsqueeze(1), embedding_labels_model.unsqueeze(0), dim=2)
        most_similar_insentence_index = similarities_word.argmax(dim=1)

        for i in range(len(most_similar_insentence_index)):
            cosine_similarity_distance.append(similarities_word[i][most_similar_insentence_index[i]])

        return most_similar_insentence_index, cosine_similarity_distance

    def predicted_word_labels_one_hot(self, y_hat):
        cosinesim_pred_word = []
        cosinesim_pred_word_cosim = []
        y_hat_prob = F.softmax(y_hat, dim=1)
        most_similar_insentence_index = torch.argmax(y_hat_prob, dim=1)
        cosine_similarity_distance = y_hat_prob[torch.arange(len(y_hat_prob)), most_similar_insentence_index]
        cosinesim_pred_word.append(most_similar_insentence_index)
        cosinesim_pred_word_cosim.append(cosine_similarity_distance)
        return cosinesim_pred_word, cosinesim_pred_word_cosim

    def training_step(self, batch):
        x, y = batch
        optimizer_g, optimizer_d = self.optimizers()
        if args.word_embedding:
            generated_vectors = self(x) # gernerate vector
            target = embedding_labels_model[y.long()] # label to vector
            MSE_loss = self.MSE_criteria(generated_vectors, target) # cal loss between generated vector and label vector
            self.log("tr_MSE_loss", MSE_loss, prog_bar=True)
            optimizer_g.zero_grad()
            self.manual_backward(MSE_loss)
            optimizer_g.step()

            if args.vec2onehot_loss:
                generated_one2vec = self(x)  # gernerate vector
                real_labels = self.discriminator(target)
                fake_labels = self.discriminator(generated_one2vec)
                real_loss = self.loss(real_labels, y.squeeze().long())
                fake_loss = self.loss(fake_labels, y.squeeze().long())
                d_loss = (real_loss + fake_loss) / 2
                self.log("tr_d_loss", d_loss, prog_bar=True)
                # d_loss.requires_grad = True
                optimizer_d.zero_grad()
                self.manual_backward(d_loss)
                optimizer_d.step()
            acc, true, preds = self.accuracy(generated_vectors, target)
        else:
            y_hat = self(x) # one-hot encoding
            loss = self.loss(y_hat, y.squeeze().long())
            optimizer_g.zero_grad()
            self.manual_backward(loss)
            optimizer_g.step()
            preds = torch.argmax(y_hat, dim=1)
            acc = torch.sum(preds == y.squeeze().long()).item() / (len(y) * 1.0)
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        if args.word_embedding:
            generated_vectors = self(x)
            target = embedding_labels_model[y.long()]
            acc, true, preds = self.accuracy(generated_vectors, target)
        else:
            y_hat = self(x)
            loss = self.loss(y_hat, y.squeeze().long())
            preds = torch.argmax(y_hat, dim=1)
            acc = torch.sum(preds == y.squeeze().long()).item() / (len(y) * 1.0)
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        if args.word_embedding:
            target = embedding_labels_model[y.long()]
            self.generated_vectors = self(x)

            loss = self.MSE_criteria(self.generated_vectors, target)
            acc, true, preds = self.accuracy(self.generated_vectors, target)
            similarities = F.cosine_similarity(self.generated_vectors.unsqueeze(1), embedding_labels_model.unsqueeze(0), dim=2)
            similarities = similarities.cpu().numpy()

            for index in range(y.shape[0]):
                similarities_matrix[y[index].long(), :] = similarities_matrix[y[index].long(), :] + similarities[index]
            preds_array = torch.stack(preds)
            test_preds.append(preds_array.squeeze())
            true_array = torch.stack(true)
            test_true.append(true_array.squeeze())
        else:
            y_hat = self(x)
            loss = self.loss(y_hat, y.squeeze().long())
            preds = torch.argmax(y_hat, dim=1)
            acc = torch.sum(preds == y.squeeze().long()).item() / (len(y) * 1.0)
            test_preds.append(preds)
            test_true.append(y.squeeze().long())

        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        global all_similarities_matrix
        if args.word_embedding:

            preds = torch.cat(test_preds)
            preds_np = preds.cpu().numpy()
            true = torch.cat(test_true)
            true_np = true.cpu().numpy()
            fold_confusion_matrix = confusion_matrix(true_np, preds_np)
            print("shape of fold_confusion_matrix", fold_confusion_matrix.shape)
            print("shape of labels", len(labels))
            if len(fold_confusion_matrix) != len(labels):
                labels_tmp = ["hey", "google", "how", "is", "the", "weather", "today", "what", "time", "it", "take",
                              "a",
                              "picture",
                              "turn", "flashlight", "on"]
                missing_c = find_missing_classes_in_fold(true_np, len(labels_tmp))
                tmp_labels = labels_tmp
                missing_c_count = 0
                for missing_number in missing_c:
                    print("missing_number", missing_number)
                    tmp_labels.pop(missing_number + missing_c_count)
                    missing_c_count = missing_c_count + 1
                BTS_confusionmatrix.plot_confusion_matrix(fold_confusion_matrix, tmp_labels,
                                                          f"{folder_name}/{args.train_name}_{fold}_all_result_confusion_matrix.png")
            else:
                BTS_confusionmatrix.plot_confusion_matrix(fold_confusion_matrix, labels,
                                                          f"{folder_name}/{args.train_name}_{fold}_all_result_confusion_matrix.png")

            # test_preds_list.append(preds_np)
            setattr(val_list, 'test_preds_list', preds_np)
            # test_true_list.append(true_np)
            setattr(val_list, 'test_true_list', true_np)
            counted_values = np.zeros(16, dtype=int)
            for value in true:
                counted_values[value] += 1
            mean_sim_matrix = similarities_matrix / counted_values[:, np.newaxis]
            # sum element wise all_similarities_matrix and mean_sim_matrix
            all_similarities_matrix = all_similarities_matrix + mean_sim_matrix
            BTS_confusionmatrix.plot_cosine_similarities_matrix(mean_sim_matrix, labels,
                                                                f"{similarity_folder}/{args.train_name}_{fold}_result_cosine_sim_matrix.png")

        else:

            preds = torch.cat(test_preds)
            preds_np = preds.cpu().numpy()
            true = torch.cat(test_true)
            true_np = true.cpu().numpy()
            fold_confusion_matrix = confusion_matrix(np.ravel(true_np), preds_np)
            print("shape of fold_confusion_matrix", fold_confusion_matrix.shape)
            print("shape of labels", len(labels))
            if len(fold_confusion_matrix) != len(labels):
                labels_tmp = labels.copy()
                missing_c = find_missing_classes_in_fold(true_np, len(labels_tmp))
                tmp_labels = labels_tmp
                missing_c_count = 0
                for missing_number in missing_c:
                    print("missing_number", missing_number)
                    tmp_labels.pop(missing_number + missing_c_count)
                    missing_c_count = missing_c_count + 1
                BTS_confusionmatrix.plot_confusion_matrix(fold_confusion_matrix, tmp_labels,
                                                          f"{folder_name}/{args.train_name}_{fold}_all_result_confusion_matrix.png")
            else:
                BTS_confusionmatrix.plot_confusion_matrix(fold_confusion_matrix, labels,
                                                          f"{folder_name}/{args.train_name}_{fold}_all_result_confusion_matrix.png")
            setattr(val_list, 'test_preds_list', preds_np)
            setattr(val_list, 'test_true_list', true_np)
            counted_values = np.zeros(16, dtype=int)
            for value in true:
                counted_values[value] += 1
            mean_sim_matrix = similarities_matrix / counted_values[:, np.newaxis]
            # sum element wise all_similarities_matrix and mean_sim_matrix
            all_similarities_matrix = all_similarities_matrix + mean_sim_matrix
            BTS_confusionmatrix.plot_cosine_similarities_matrix(mean_sim_matrix, labels,
                                                                f"{similarity_folder}/{args.train_name}_{fold}_result_softmax_matrix.png")

        if args.included_testset:
            WIS_true_word = [labels[idx.item()] for idx in true_np]
            WIS_pred_word = [labels[idx] for idx in preds_np]
            setattr(val_list, 'WIS_true_list', WIS_true_word)
            setattr(val_list, 'WIS_pred_list', WIS_pred_word)
            num_correct = 0
            for i in range(len(WIS_pred_word)):
                if WIS_pred_word[i] == WIS_true_word[i]:
                    num_correct += 1
            WER = num_correct / len(WIS_pred_word)
            setattr(val_list, 'WIS_WER_list', WER)

            if args.included_intestset:
                splited_true_WIS = BTS_utils.split_sentence_WIS_true_labels(WIS_true_word, start_word)

                splited_pred_WIS = BTS_utils.split_sentence_WIS_pred_labels(WIS_pred_word, splited_true_WIS)
                corrected_splited_pred_WIS_list = []
                for i in range(len(splited_pred_WIS)):
                    corrected_sen_prob, corrected_splited_pred_WIS = BTS_LM.BTS_viterbi(splited_pred_WIS[i])
                    corrected_splited_pred_WIS_list.append(corrected_splited_pred_WIS)
                corrected_pred_WIS = [tmp_word for tmp_label in corrected_splited_pred_WIS_list for tmp_word in
                                      tmp_label]
                for WIS_i in range(len(corrected_splited_pred_WIS_list)):
                    corrected_selected_sentence = BTS_utils.find_matching_sentence(
                        ' '.join(corrected_splited_pred_WIS_list[WIS_i]), sentence_labels)
                    WIS_selected_sentence = BTS_utils.find_matching_sentence(' '.join(splited_pred_WIS[WIS_i]),
                                                                             sentence_labels)
                    setattr(val_list, 'WIS_pred_sen_list', WIS_selected_sentence)
                    setattr(val_list, 'WIS_corrected_pred_sen_list', corrected_selected_sentence)
                    val_list.add_to_list('WIS_true_sen_list', ' '.join(splited_true_WIS[WIS_i]))

                setattr(val_list, 'WIS_corrected_pred_list', corrected_pred_WIS)
                num_correct_nlp = 0
                for i in range(len(corrected_pred_WIS)):
                    if corrected_pred_WIS[i] == WIS_true_word[i]:
                        num_correct_nlp += 1
                WER_WIS_tmp = num_correct_nlp / len(corrected_pred_WIS)
                setattr(val_list, 'WIS_WER_NLP_list', WER_WIS_tmp)

                print('WER_WIS_tmp', WER_WIS_tmp)
        else:
            pass

    def predict_step(self, batch, batch_idx):
        x, y = batch
        generated_vectors = self(x)

        if args.word_embedding:
            predicted_word_label, predicted_word_similarity = self.predicted_word_labels(generated_vectors)

            preds_word_index = predicted_word_label
            setattr(val_list, 'pred_step_word_index', preds_word_index)
            sentence_array = torch.squeeze(y)
            setattr(val_list, 'pred_step_sentence_index', sentence_array)
            preds_word_similarity = torch.stack(predicted_word_similarity)
            setattr(val_list, 'pred_step_word_similarity', preds_word_similarity)
        else:
            predicted_word_label, predicted_word_similarity = self.predicted_word_labels_one_hot(generated_vectors)
            preds_word_index = predicted_word_label
            preds_word_index = torch.cat(preds_word_index, dim=0)
            setattr(val_list, 'pred_step_word_index', preds_word_index)
            sentence_array = torch.squeeze(y)
            setattr(val_list, 'pred_step_sentence_index', sentence_array)
            preds_word_similarity = predicted_word_similarity
            preds_word_similarity = torch.cat(preds_word_similarity, dim=0)
            setattr(val_list, 'pred_step_word_similarity', preds_word_similarity)
        return predicted_word_label

    def on_predict_end(self):

        preds_pred_step = torch.cat(getattr(val_list, "pred_step_word_index"))
        preds_pred_step_np = preds_pred_step.cpu().numpy()
        similarity_pred_step = torch.cat(getattr(val_list, "pred_step_word_similarity"))

        similarity_pred_step_np = similarity_pred_step.cpu().numpy()

        threshord_indices = np.where(similarity_pred_step_np > 0.5)[0]
        prob_preds = preds_pred_step_np[threshord_indices]
        true_pred_step = torch.cat(getattr(val_list, "pred_step_sentence_index"))
        true_pred_step_np = true_pred_step.cpu().numpy()
        true_pred_step_np = true_pred_step_np.astype(int)
        WIS_sen_true_word = [sentence_labels[idx.item()] for idx in true_pred_step_np]
        WIS_sen_pred_word = [labels[idx] for idx in prob_preds]
        selected_sentence = BTS_utils.find_matching_sentence(' '.join(WIS_sen_pred_word), sentence_labels)
        setattr(val_list, 'fold_true_sen_list', WIS_sen_true_word[0])
        setattr(val_list, 'fold_pred_sen_list', selected_sentence)

        if selected_sentence == WIS_sen_true_word[0]:
            sen_correct = 1
        else:
            sen_correct = 0
        setattr(val_list, 'fold_sen_SER_list', sen_correct)
        setattr(val_list, 'fold_pred_word_sen_list', WIS_sen_pred_word)
        setattr(val_list, 'fold_similar_sen_list', similarity_pred_step_np)

        val_list.pred_step_word_index.clear()
        val_list.pred_step_sentence_index.clear()
        val_list.pred_step_word_similarity.clear()


    def configure_optimizers(self):

        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        scheduler_g = CosineAnnealingLR(optimizer_g, T_max=400, eta_min=1 * 1e-5)
        #

        optimizer_d = torch.optim.Adam(list(self.generator.parameters())+list(self.discriminator.parameters()), lr=1e-4)
        return [optimizer_g, optimizer_d], [scheduler_g]


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("running validation...")
        return bar


def find_missing_classes_in_fold(true_arr, num_unique_labels):
    numbers = np.arange(num_unique_labels)
    unique_elements = np.unique(true_arr)
    missing_numbers = np.setdiff1d(numbers, unique_elements)
    return list(missing_numbers)


def lightning_train(train, val, testset, x_sentence_data, y_sentence_data, fold, args):
    model = seq2seq_Model()

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{folder_name}/models/fold_{fold}",
        filename=f'{args.train_name}_fold_{fold}_{{epoch:02d}}_{{val_acc:.4f}}',
        # filename=f'{args.train_name}_fold_{fold}_{{val_loss:.4f}}_{{epoch:02d}}_{{val_acc:.4f}}',
        save_top_k=3,
        verbose=True,
        save_last=False,
        monitor='val_acc',
        mode='max',
        save_on_train_epoch_end=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,

    )
    trainer.fit(model, train, val)

    if trainer.checkpoint_callback.best_model_path is not None:
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f'Fold {fold}: Best model saved at {best_model_path}')
    else:
        print(f'Fold {fold}: No best model saved')

    best_model_val_acc_path = BTS_utils.load_best_val_acc_model(checkpoint_callback.dirpath)
    best_model = seq2seq_Model.load_from_checkpoint(best_model_val_acc_path)
    print("best_model_name", best_model_val_acc_path)

    test_result = trainer.test(best_model, testset)
    if args.predict_sentence:
        print('here')
        for i in range(len(y_sentence_data)):
            x_aug_sentence_data, y_aug_sentence_data = BTS_dataset.sentence_windowing(
                x_sentence_data[i, :, 0:4000][np.newaxis, :, :], y_sentence_data[i], args.n_time)
            X_predict = torch.tensor(x_aug_sentence_data, dtype=torch.float32, device=device)
            Y_predict = torch.tensor(y_aug_sentence_data, dtype=torch.float32, device=device)
            predict_dataset = TensorDataset(X_predict, Y_predict)
            predset = torch.utils.data.DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False)
            prediction_result = trainer.predict(best_model, predset)

        setattr(val_list, 'sen_SER_list',
                np.sum(getattr(val_list, 'fold_sen_SER_list').copy()) / len(getattr(val_list, 'fold_sen_SER_list').copy()))
        # sen_SER_list.append(SER_fold)
        setattr(val_list, 'true_sen_list', getattr(val_list,"fold_true_sen_list").copy())
        # true_sen_list.append(fold_true_sen_list)
        setattr(val_list, 'pred_sen_list', getattr(val_list,"fold_pred_sen_list").copy())
        # pred_sen_list.append(fold_pred_sen_list)
        setattr(val_list, 'pred_word_sen_list', getattr(val_list,"fold_pred_word_sen_list").copy())
        # pred_word_sen_list.append(fold_pred_word_sen_list)
        setattr(val_list, 'pred_similarity_sen_list', getattr(val_list,"fold_similar_sen_list").copy())
        # pred_similarity_sen_list.append(fold_similar_sen_list)

        val_list.fold_sen_SER_list.clear()
        val_list.fold_true_sen_list.clear()
        val_list.fold_pred_sen_list.clear()
        val_list.fold_pred_word_sen_list.clear()
        val_list.fold_similar_sen_list.clear()

    # test_accs.append(test_result[0]['test_acc_epoch'])
    setattr(val_list, 'test_accs', test_result[0]['test_acc_epoch'])

    print(f'Fold {fold}: Test accuracy: {test_result[0]["test_acc_epoch"]:.4f}')


def set_seed(seed=42, determine=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if determine:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    set_seed(42, determine=False)

    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='BTS_demo')
    parser.add_argument('--data-root',default='/media/easyonemain/새 볼륨/semantic_data/1_preprocessing_30_120_EOG_basec/Datafiles/Demo_data/')
    parser.add_argument('--result-index-root', default='/media/easyonemain/새 볼륨/BTS_demo_result/good_index/')
    parser.add_argument('--embedding-labels-path', type=str,default='embedding_vector/demo_embeddingvector_16labels_20d_mi.npy')
    parser.add_argument('--save_path', type=str, default='/media/easyonemain/새 볼륨/BTS_demo_result')
    parser.add_argument('--epochs', type=int, default=300, metavar='N')
    parser.add_argument('--CAR', type=bool, default=True)
    parser.add_argument('--Zscore', type=bool, default=False)
    parser.add_argument('--windowing', type=bool, default=False)
    parser.add_argument('--twodimconv', type=bool, default=False)
    parser.add_argument('--word-embedding', type=bool, default=True)
    parser.add_argument('--vec2onehot-loss', type=bool, default=False)
    parser.add_argument('--included-testset', type=bool, default=True)
    parser.add_argument('--included-intestset', type=bool, default=True)
    parser.add_argument('--predict-sentence', type=bool, default=True)
    parser.add_argument('--k-fold-CV', type=int, default=10, metavar='N')
    parser.add_argument('--overlap-size', type=int, default=100, metavar='N')
    parser.add_argument('--window-size', type=int, default=1000, metavar='N')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N')
    parser.add_argument('--task-selected', type=str, default='os')
    parser.add_argument('--sub-list', nargs='+', type=int, default=[1])
    args = parser.parse_args()
    data_file_name_list = [f'sub_{sub_num}_{args.task_selected}_30_120_eog_bp.mat' for sub_num in args.sub_list]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for sub in range(1, len(args.sub_list)+1):
        val_list = BTS_utils.Val_list()
        embedding_labels_model = torch.from_numpy(np.load(args.embedding_labels_path) * 10).float().to(device)
        if args.word_embedding:
            vecorone_task = 'embedding_vector'
            args.vec_dim = embedding_labels_model.shape[1]
        else:
            vecorone_task = 'one_hot'
            args.vec_dim = embedding_labels_model.shape[0]
        args.data_file_name = data_file_name_list[sub - 1]
        print(args.data_file_name)
        if args.word_embedding:
            args.train_name = f'sub{args.sub_list[sub - 1]}_{args.task_selected}_k5_10fold_nn_{vecorone_task}_CAR_GAN_{args.vec_dim}d_the_test'
        else:
            args.train_name = f'sub{args.sub_list[sub - 1]}_{args.task_selected}_k5_10fold_nn_{vecorone_task}_onehot'

        x_data, y_data, y_origin, x_test_data, y_test_data, y_test_origin, x_sentence_data, y_sentence_data = BTS_dataset.input_Neural_vector(args)
        all_similarities_matrix = np.zeros((len(np.unique(y_origin)), len(np.unique(y_origin))))

        folder_name, similarity_folder = BTS_utils.make_save_folder(args, f"{args.save_path}/{args.train_name}_test")
        labels = ["hey", "google", "how", "is", "the", "weather", "today", "what", "time", "it", "take",
                  "a", "picture", "turn", "flashlight", "on"]
        sentence_labels = ["hey google", "how is the weather today", "what time is it", "take a picture",
                           "turn the flashlight on"]
        start_word = [sentence.split()[0] for sentence in sentence_labels]

        if args.included_testset:
            print("inference WIS and natural sentence generation")
            train_indices_list = []
            val_indices_list = []
            test_indices_list = []
            num_test_trials = 5
            sub_best_test_index, sub_tr_index = BTS_utils.load_best_index(args, sub)
            for fold in range(num_test_trials):
                print(f'Fold {fold}:')
                test_preds = []
                test_preds_e = []
                test_true = []
                similarities_matrix = np.zeros((len(np.unique(y_origin)), len(np.unique(y_origin))))

                tr_WIS_indices = [0, 1, 2, 3, 4]
                tr_WIS_indices.remove((fold % 5))
                tr_WIS_indices.remove(((fold + 1) % 5))
                train_loader, valid_loader, test_loader = BTS_dataset.WIS_NS_dataset(x_data, y_origin,x_test_data,y_test_origin, sub_best_test_index, sub_tr_index, fold, args, device)
                lightning_train(train_loader, valid_loader, test_loader, x_sentence_data, y_sentence_data, fold,
                                args)
        else:
            print("Baseline with K-fold CV")
            skf = StratifiedKFold(n_splits=args.k_fold_CV, shuffle=True, random_state=42)
            for fold, (train_index, validation_index) in enumerate(skf.split(x_data, y_origin)):
                test_preds = []
                test_preds_e = []
                test_true = []
                similarities_matrix = np.zeros((len(np.unique(y_origin)), len(np.unique(y_origin))))
                print(f'Fold {fold}:')
                train_loader, valid_loader, test_loader = BTS_dataset.k_fold_dataset(x_data, y_origin, train_index, validation_index, fold, args, device, val_list)
                lightning_train(train_loader, valid_loader, test_loader, x_sentence_data, y_sentence_data, fold, args)

        print(f'Average test accuracy: {np.mean(getattr(val_list, "test_accs")):.4f}')
        # label_list = labels.copy()
        all_similarities_matrix = all_similarities_matrix / args.k_fold_CV
        tmp = (np.ones((5, 5)) * 5) / args.k_fold_CV
        BTS_confusionmatrix.plot_cosine_similarities_matrix(all_similarities_matrix, labels,
                                                            f"{similarity_folder}/{args.train_name}_all_result_cosine_sim_matrix.png")

        all_preds = np.concatenate(getattr(val_list, "test_preds_list"))
        all_trues = np.concatenate(getattr(val_list, "test_true_list"))
        WIS_SER_num = []
        WIS_corrected_SER_num = []
        for WIS_SER_i in range(len(getattr(val_list, "WIS_true_sen_list"))):
            if getattr(val_list, "WIS_true_sen_list")[WIS_SER_i] == getattr(val_list, "WIS_pred_sen_list")[WIS_SER_i]:
                sen_correct = 1
            else:
                sen_correct = 0
            if getattr(val_list, "WIS_true_sen_list")[WIS_SER_i] == getattr(val_list, "WIS_corrected_pred_sen_list")[WIS_SER_i]:
                sen_corrected = 1
            else:
                sen_corrected = 0
            WIS_SER_num.append(sen_correct)
            WIS_corrected_SER_num.append(sen_corrected)
        np.array(WIS_SER_num)
        WIS_SER = np.sum(WIS_SER_num) / len(WIS_SER_num)
        np.array(WIS_corrected_SER_num)
        WIS_corrected_SER = np.sum(WIS_corrected_SER_num) / len(WIS_corrected_SER_num)
        WIS_SER = np.full((len(getattr(val_list, "sen_SER_list")),), WIS_SER)
        WIS_corrected_SER = np.full((len(getattr(val_list, "sen_SER_list")),), WIS_corrected_SER)
        sub_number = args.sub_list[sub - 1]
        # X_train, X_val = x_data[getattr(val_list, "train_indices_list")[fold]], x_data[
        #     getattr(val_list, "val_indices_list")[fold]]
        if args.included_testset:
            pass
        else:
            best_fold_index = np.argmax(getattr(val_list, "test_accs"))
            if args.word_embedding:
                np.savez(f"{args.result_index_root}/sub{args.sub_list[sub - 1]}_{args.task_selected}_fold_good_index_embedding_vector.npz", train_indices_list=getattr(val_list, "train_indices_list"), val_indices_list=getattr(val_list, "val_indices_list"),
                         test_indices_list=getattr(val_list, "test_indices_list"), best_train_indices=getattr(val_list, "train_indices_list")[best_fold_index], best_val_indices=getattr(val_list, "val_indices_list")[best_fold_index], best_test_indices=getattr(val_list, "test_indices_list")[best_fold_index])
            else:
                np.savez(f"{args.result_index_root}/sub{args.sub_list[sub - 1]}_{args.task_selected}_fold_good_index_one_hot.npz", train_indices_list=getattr(val_list, "train_indices_list"), val_indices_list=getattr(val_list, "val_indices_list"),
                         test_indices_list=getattr(val_list, "test_indices_list"), best_train_indices=getattr(val_list, "train_indices_list")[best_fold_index], best_val_indices=getattr(val_list, "val_indices_list")[best_fold_index], best_test_indices=getattr(val_list, "test_indices_list")[best_fold_index])

        if args.included_testset:
            if args.included_intestset:
                BTS_utils.save_results(folder_name, args.train_name, all_preds, all_trues, getattr(val_list, "test_accs"), num_test_trials, labels,
                                       args, getattr(val_list, "test_preds_list"), getattr(val_list, "test_true_list"), getattr(val_list, "WIS_WER_list"), getattr(val_list, "WIS_WER_NLP_list"),
                                       getattr(val_list, "WIS_pred_list"), getattr(val_list, "WIS_corrected_pred_list"), getattr(val_list, "WIS_true_list"), getattr(val_list, "WIS_pred_sen_list"),
                                       getattr(val_list, "WIS_corrected_pred_sen_list"), getattr(val_list, "WIS_true_sen_list"), WIS_SER, WIS_corrected_SER,
                                       getattr(val_list, "sen_SER_list"), getattr(val_list, "true_sen_list"), getattr(val_list, "pred_sen_list"), getattr(val_list, "pred_word_sen_list"),
                                       getattr(val_list, "pred_similarity_sen_list"))
        else:
            BTS_utils.save_results_k_fold(folder_name, args.train_name, all_preds, all_trues, getattr(val_list, "test_accs"), args.k_fold_CV,
                                          labels, args, getattr(val_list, "test_preds_list"), getattr(val_list, "test_true_list"))
        BTS_utils.process_confusion_matrices(args.k_fold_CV, folder_name, args.train_name, all_trues, all_preds, labels,
                                             args)
