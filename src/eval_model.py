import torch
from torch.utils.data import DataLoader
from dataset import MeteoDataset
from utils import *
from sampler import *
import os
import argparse
from tqdm import tqdm

from models.traj_gru import TrajGRU
from models.conv_gru import ConvGRU
from models.naive_cnn import cnn_2D
from models.u_net import UNet

def main(path, device, epoch, save_preds):

    checkpoint = torch.load(path + '/model_{}.pth'.format(epoch), map_location=device)
    network = TrajGRU(device=device)
    network.to(device)
    network.load_state_dict(checkpoint.state_dict())
    network.eval()

    if not os.path.isdir(path + '/images'):
        os.mkdir(path + '/images')

    input_length=12
    output_length=12
    batch_size=4

    test = MeteoDataset(rain_dir='../../Data/MeteoNet-Brest/rainmap/val',
                        wind_dir='../../Data/MeteoNet-Brest/wind',
                        input_length=input_length,
                        output_length=output_length,
                        temporal_stride=12,
                        dataset='test',
                        recurrent_nn=True)

    test_sampler = CustomSampler(indices_except_undefined_sampler(test, recurrent_nn, wind), test, wind=True)
    n_examples_test = len(test)
    test_dataloader = DataLoader(test, batch_size=batch_size, sampler=test_sampler)

    print("Len_dataloader_test : ", len(test_dataloader)*batch_size)

    thresholds = np.array([0.1, 1, 2.5])

    test_loss = 0.0

    index_plot = 0

    confusion_matrix = {}
    for thresh in thresholds:
        confusion_matrix[str(thresh)] = {'true_positive' : [0]*output_length, 'true_negative' : [0]*output_length,
        'false_positive' : [0]*output_length, 'false_negative' : [0]*output_length}

    for sample in tqdm(test_dataloader):
        inputs, targets = sample['input'], sample['target']
        inputs = inputs.to(device=device)
        targets = targets.to(device=device)
        outputs = network(inputs)
        mask = compute_weight_mask(targets)
        loss = weighted_mse_loss(outputs, targets, mask) + weighted_mae_loss(outputs, targets, mask)
        test_loss += loss.item() / n_examples_test

        for thresh in thresholds:
            conf_mat_batch = compute_confusion_matrix_on_batch(outputs, targets, thresh)
            confusion_matrix = add_confusion_matrix_on_batch(confusion_matrix, conf_mat_batch, thresh)

        for k in range(inputs.shape[0]):
            plot_output_gt(outputs[k], targets[k], inputs[k], index=index_plot, output_dir=path+'/images/')
            save_gif(outputs[k], path+'/images/pred_{}.gif'.format(index_plot))
            save_gif(targets[k], path+'/images/target_{}.gif'.format(index_plot))
            index_plot += 1

    scores_evaluation = model_evaluation(confusion_matrix)

    print(f"[Test] Loss : {test_loss:.2f}")
    print("[Test] metrics_scores : ", scores_evaluation)
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='The directory of the model')
    parser.add_argument('--cuda', action='store_true', help='If we want to use cuda')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--save_preds', action='store_true')
    args = parser.parse_args()

    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("Cuda device not available")
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    main(args.model_path, device, args.epoch, args.save_preds)
