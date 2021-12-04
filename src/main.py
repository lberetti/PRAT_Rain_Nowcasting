import torch
import argparse
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import cnn_2D
from dataset import MeteoDataset
from utils import *


def train_network(network, input_length, output_length, epochs, batch_size, device, log_dir, save_pred=False, print_metric_logs=False):

    writer = SummaryWriter(log_dir)

    train = MeteoDataset(rain_dir='../../Data/MeteoNet-Brest/rainmap/train', input_length=input_length,  output_length=output_length)
    val = MeteoDataset(rain_dir='../../Data/MeteoNet-Brest/rainmap/val', input_length=input_length, output_length=output_length)

    n_examples_train = len(train)
    n_examples_valid = len(val)

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)


    optimizer = torch.optim.Adam(network.parameters(), lr=10**-4)
    #criterion = torch.nn.MSELoss()

    thresholds = [0.5, 2, 5, 10, 30]
    metrics = build_metrics_dict(thresholds)


    for epoch in range(epochs):

        network.train()
        training_loss = 0.0
        validation_loss = 0.0

        loop = tqdm(train_dataloader)
        loop.set_description(f"Epoch {epoch+1}/{epochs}")

        for key in metrics['train']:
            metrics['train'][key].on_epoch_begin()
            metrics['val'][key].on_epoch_begin()

        for batch_idx, sample in enumerate(loop):
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()
            outputs = network(inputs)
            mask = compute_weight_mask(targets)
            loss = weighted_mse_loss(outputs, targets, mask) + weighted_mae_loss(outputs, targets, mask)
            #loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() / n_examples_train

            loop.set_postfix({'Train Loss' : training_loss})

            for key in metrics['train']:
                metrics['train'][key].compute_metric(outputs, targets)

        for key in metrics['train']:
            metrics['train'][key].on_epoch_end()
        metrics['train'] = round_dictionnary_values(metrics['train'], 3)

        if print_metric_logs:
            print("[Train] metrics_scores : ", {k: v.total for k, v in metrics['train'].items()})

        network.eval()

        for sample in valid_dataloader:
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = network(inputs)
            mask = compute_weight_mask(targets)
            loss = weighted_mse_loss(outputs, targets, mask) + weighted_mae_loss(outputs, targets, mask)
            #loss = criterion(outputs, targets)
            validation_loss += loss.item() / n_examples_valid

            for key in metrics['val']:
                metrics['val'][key].compute_metric(outputs, targets)

        for key in metrics['val']:
            metrics['val'][key].on_epoch_end()
        metrics['val'] = round_dictionnary_values(metrics['val'], 3)

        print(f"[Validation] Loss : {validation_loss:.2f}")
        if print_metric_logs:
            print("[Validation] metrics_scores : ", {k: v.total for k, v in metrics['val'].items()})
        print("\n")

        writer.add_scalar('Loss/train', training_loss, epoch)
        writer.add_scalar('Loss/test', validation_loss, epoch)

        for key in metrics['train']:
            writer.add_scalar(key + '/train', metrics['train'][key].total, epoch)
        for key in metrics['val']:
            writer.add_scalar(key + '/valid', metrics['val'][key].total, epoch)


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

    args = parser.parse_args()

    log_dir = './runs/epochs_{}_batch_size_{}_IL_{}_OL_{}'.format(args.epochs, args.batch_size, args.input_length, args.output_length)
    if os.path.isdir(log_dir):
        raise Exception("Path {} already exists".format(log_dir))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    network = cnn_2D(input_length=args.input_length, output_length=args.output_length, filter_number=70)
    summary(network, input_size=(12, 128, 128), device='cpu')
    network.to(device=device)
    train_network(network, input_length=args.input_length,
                    output_length=args.output_length,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                    log_dir=log_dir,
                    print_metric_logs=args.print_metric_logs
                    )
