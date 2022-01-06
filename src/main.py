import torch
import argparse
#from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.u_net import UNet
from models.naive_cnn import cnn_2D
from models.traj_gru import TrajGRU
from dataset import MeteoDataset
from utils import *


def train_network(network, input_length, output_length, epochs, batch_size, device, log_dir, save_pred=False, print_metric_logs=False, recurrent_nn=False):

    writer = SummaryWriter(log_dir)

    train = MeteoDataset(rain_dir='../../Data/MeteoNet-Brest/rainmap/train',
                        input_length=input_length,
                        output_length=output_length,
                        temporal_stride=input_length,
                        dataset='train',
                        recurrent_nn=recurrent_nn)
    train_sampler = CustomSampler(indices_except_undefined_sampler(train), train)
    val = MeteoDataset(rain_dir='../../Data/MeteoNet-Brest/rainmap/val',
                        input_length=input_length,
                        output_length=output_length,
                        temporal_stride=input_length,
                        dataset='valid',
                        recurrent_nn=recurrent_nn)
    val_sampler = CustomSampler(indices_except_undefined_sampler(val), val)

    print("Len train dataset : ", len(train))
    print("Len val dataset : ", len(val))

    n_examples_train = len(train)
    n_examples_valid = len(val)

    train_dataloader = DataLoader(train, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(val, batch_size=batch_size, sampler=val_sampler)

    print("Len_dataloader_train : ", len(train_dataloader))
    print("Len_dataloader_valid : ", len(valid_dataloader))

    lr = 10**-3
    wd = 0.1
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=wd)
    #criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    thresholds = [0.1, 1, 2.5]

    info = f'''Starting training:
        Epochs:                {epochs},
        Learning rate:         {lr},
        Batch size:            {batch_size},
        Weight decay:          {wd},
        Number batch train :   {len(train_dataloader)},
        Number batch val :     {len(valid_dataloader)},
        Scheduler :            Gamma 0.1 epochs 10, 20, 30
    '''
    writer.add_text('Description', info)


    for epoch in range(epochs):

        """if epoch > 10:
            for g in optimizer.param_groups:
                g['lr'] = 10**-5
        if epoch > 20:
            for g in optimizer.param_groups:
                g['lr'] = 10**-6"""

        confusion_matrix = {}
        for thresh in thresholds:
            confusion_matrix[str(thresh)] = {'true_positive' : [0]*output_length, 'true_negative' : [0]*output_length,
            'false_positive' : [0]*output_length, 'false_negative' : [0]*output_length}

        network.train()
        training_loss = 0.0
        validation_loss = 0.0

        loop = tqdm(train_dataloader)
        loop.set_description(f"Epoch {epoch+1}/{epochs}")

        writer.add_scalar('LR : ', scheduler.get_last_lr()[0], epoch)

        for batch_idx, sample in enumerate(loop):
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()
            outputs = network(inputs)
            mask = compute_weight_mask(targets)
            loss = 0.00005*(weighted_mse_loss(outputs, targets, mask) + weighted_mae_loss(outputs, targets, mask))
            #loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() / n_examples_train

            #print("Network weight i2f : ", network.traj_gru_5.i2f.weight)
            #print("Network grad i2f : ", network.traj_gru_5.i2f.weight.grad)
            #print("Network weight h2f : ", network.traj_gru_5.h2f.weight)
            #print("Network grad h2f : ", network.traj_gru_5.h2f.weight.grad)
            #print("Network weight flows : ", network.traj_gru_5.flows.weight)
            #print("Network grad flows : ", network.traj_gru_5.flows.weight.grad)

            loop.set_postfix({'Train Loss' : training_loss})

        scheduler.step()

        network.eval()

        for sample in valid_dataloader:
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = network(inputs)
            mask = compute_weight_mask(targets)
            loss = 0.00005*(weighted_mse_loss(outputs, targets, mask) + weighted_mae_loss(outputs, targets, mask))
            #loss = criterion(outputs, targets)
            validation_loss += loss.item() / n_examples_valid

            for thresh in thresholds:
                conf_mat_batch = compute_confusion_matrix_on_batch(outputs, targets, thresh)
                confusion_matrix = add_confusion_matrix_on_batch(confusion_matrix, conf_mat_batch, thresh)


        writer.add_scalar('Loss/train', training_loss, epoch)
        writer.add_scalar('Loss/test', validation_loss, epoch)
        print(f"[Validation] Loss : {validation_loss:.2f}")

        if epoch > 4:
            scores_evaluation = model_evaluation(confusion_matrix)

            if print_metric_logs:
                print("[Validation] metrics_scores : ", scores_evaluation)
            print("\n")


            for thresh_key in scores_evaluation:
                for metric_key in scores_evaluation[thresh_key]:
                    for time_step in scores_evaluation[thresh_key][metric_key]:
                        writer.add_scalar(metric_key + "_" + thresh_key + "_time_step_" + time_step,
                                        scores_evaluation[thresh_key][metric_key][time_step],
                                        epoch)

        """torch.save({'epoch' : epoch + 1,
                    'model_state_dict' : network.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict()},
                    log_dir + '/model_{}.pth'.format(epoch+1))"""
        torch.save(network, log_dir + '/model_{}.pth'.format(epoch+1))

    if save_pred:
        os.mkdir(log_dir + '/images')
        save_pred_images(network, val, n_plots=30, output_dir=log_dir + '/images/', device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help="The number of epochs used to train the network")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--input_length', type=int, default=12, help="The number of time steps of a sequence as input of the NN")
    parser.add_argument('--output_length', type=int, default=5, help="The number of time steps predicted by the NN")
    parser.add_argument('--save_preds', action='store_true', help='If we want to save some predictions according to the ground truth.')
    parser.add_argument('--print_metric_logs', action='store_true', help='If we want to print the metrics score while training')
    parser.add_argument('--recurrent_nn', action='store_true', help='If the model to train is recurrent')

    args = parser.parse_args()

    log_dir = './runs/epochs_{}_batch_size_{}_IL_{}_OL_{}'.format(args.epochs, args.batch_size, args.input_length, args.output_length)
    if os.path.isdir(log_dir):
        raise Exception("Path {} already exists".format(log_dir))


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    device = torch.device('cuda:0')
    print(f'Using device {device}')
    network = TrajGRU(device=device)
    #network = cnn_2D(input_length=args.input_length, output_length=args.output_length, filter_number=16)
    #network = UNet(input_length=args.input_length, output_length=args.output_length, filter_number=16)
    #summary(network, input_size=(12, 128, 128), device='cpu')
    network.to(device=device)
    train_network(network, input_length=args.input_length,
                    output_length=args.output_length,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                    log_dir=log_dir,
                    save_pred=args.save_preds,
                    print_metric_logs=args.print_metric_logs,
                    recurrent_nn=args.recurrent_nn
                    )
