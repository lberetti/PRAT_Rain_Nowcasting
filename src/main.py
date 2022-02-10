import torch
import argparse
#from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.u_net import UNet
from models.naive_cnn import cnn_2D
from models.traj_gru import TrajGRU
from models.conv_gru import ConvGRU
from dataset import MeteoDataset
from utils import *
from sampler import *


def train_network(network, input_length, output_length, epochs, batch_size, device, log_dir, print_metric_logs=False, recurrent_nn=False, wind=False):

    writer = SummaryWriter(log_dir)

    if wind:
        wind_dir = '../../Data/MeteoNet-Brest/wind'
    else:
        wind_dir = None

    print("Building the train dataset ...")

    train = MeteoDataset(rain_dir='../../Data/MeteoNet-Brest/rainmap/train',
                        wind_dir=wind_dir,
                        input_length=input_length,
                        output_length=output_length,
                        temporal_stride=input_length,
                        dataset='train',
                        recurrent_nn=recurrent_nn)

    print("Sampling train dataset ...")
    train_sampler = CustomSampler(indices_except_undefined_sampler(train, recurrent_nn, wind), train)

    print("Building validation dataset ...")
    val = MeteoDataset(rain_dir='../../Data/MeteoNet-Brest/rainmap/val',
                        wind_dir=wind_dir,
                        input_length=input_length,
                        output_length=output_length,
                        temporal_stride=input_length,
                        dataset='valid',
                        recurrent_nn=recurrent_nn)

    print("Sampling validation dataset ...")
    val_sampler = CustomSampler(indices_except_undefined_sampler(val, recurrent_nn, wind), val)

    print("Len train dataset : ", len(train))
    print("Len val dataset : ", len(val))

    n_examples_train = len(train)
    n_examples_valid = len(val)


    train_dataloader = DataLoader(train, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(val, batch_size=batch_size, sampler=val_sampler)

    print("Len_dataloader_train : ", len(train_dataloader))
    print("Len_dataloader_valid : ", len(valid_dataloader))

    lr = 10**-5
    wd = 0.1
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    thresholds = [0.1, 1, 2.5]

    info = f'''Starting training:
        Epochs:                {epochs},
        Learning rate:         {lr},
        Batch size:            {batch_size},
        Weight decay:          {wd},
        Number batch train :   {len(train_dataloader)},
        Number batch val :     {len(valid_dataloader)},
        Scheduler :            Gamma 0.1 epochs 30, 60
    '''
    writer.add_text('Description', info)


    for epoch in range(epochs):

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
            loss = 0.0005*(weighted_mse_loss(outputs, targets, mask) + weighted_mae_loss(outputs, targets, mask))
            loss.backward()
            optimizer.step()

            training_loss += loss.item() / n_examples_train

            loop.set_postfix({'Train Loss' : training_loss})

        scheduler.step()

        network.eval()

        for sample in valid_dataloader:
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = network(inputs)
            mask = compute_weight_mask(targets)
            loss = 0.0005*(weighted_mse_loss(outputs, targets, mask) + weighted_mae_loss(outputs, targets, mask))
            validation_loss += loss.item() / n_examples_valid

            for thresh in thresholds:
                conf_mat_batch = compute_confusion_matrix_on_batch(outputs, targets, thresh)
                confusion_matrix = add_confusion_matrix_on_batch(confusion_matrix, conf_mat_batch, thresh)


        writer.add_scalar('Loss/train', training_loss, epoch)
        writer.add_scalar('Loss/test', validation_loss, epoch)
        print(f"[Validation] Loss : {validation_loss:.2f}")

        # Compute after epoch 4 to prevent division by 0 in some metrics
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

        torch.save(network, log_dir + '/model_{}.pth'.format(epoch+1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help="The number of epochs used to train the network")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--input_length', type=int, default=12, help="The number of time steps of a sequence as input of the NN")
    parser.add_argument('--output_length', type=int, default=12, help="The number of time steps predicted by the NN")
    parser.add_argument('--print_metric_logs', action='store_true', help='If we want to print the metrics score while training')
    parser.add_argument('--wind', action='store_true', help="If we want to use the wind")
    parser.add_argument('--network', choices=['TrajGRU', 'ConvGRU', 'CNN2D'])
    args = parser.parse_args()

    log_dir = './runs/epochs_{}_batch_size_{}_IL_{}_OL_{}'.format(args.epochs, args.batch_size, args.input_length, args.output_length)
    if os.path.isdir(log_dir):
        raise Exception("Path {} already exists".format(log_dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    if args.network == 'ConvGRU':
        recurrent_nn = True
        network = ConvGRU(device=device, wind=args.wind)
    elif args.network == 'TrajGRU':
        recurrent_nn = True
        network = TrajGRU(device=device, wind=args.wind)
    elif args.network == 'CNN2D':
        recurrent_nn = False
        network = cnn_2D(input_length=args.input_length, output_length=args.output_length, filter_number=16, wind=args.wind)

    network.to(device=device)

    train_network(network, input_length=args.input_length,
                    output_length=args.output_length,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                    log_dir=log_dir,
                    print_metric_logs=args.print_metric_logs,
                    recurrent_nn=recurrent_nn,
                    wind=args.wind
                    )
