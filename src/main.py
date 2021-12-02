import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import cnn_2D
from dataset import MeteoDataset
from utils import *


def train_network(network, input_length, output_length, epochs, batch_size, device):

    writer = SummaryWriter()

    train = MeteoDataset(rain_dir='../../Data/MeteoNet-Brest/rainmap/train', input_length=input_length,  output_length=output_length)
    val = MeteoDataset(rain_dir='../../Data/MeteoNet-Brest/rainmap/val', input_length=input_length, output_length=output_length)

    n_examples_train = len(train)
    n_examples_valid = len(val)

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)


    optimizer = torch.optim.Adam(network.parameters(), lr=10**-4)
    #criterion = torch.nn.MSELoss()

    for epoch in range(epochs):

        network.train()
        training_loss = 0.0
        validation_loss = 0.0

        # TODO : Create a class for metrics
        csi_scores_train = {}
        csi_scores_val = {}
        csi_seen_train = {}
        csi_seen_val = {}

        loop = tqdm(train_dataloader)
        loop.set_description(f"Epoch {epoch+1}/{epochs}")

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

            csi_scores = CSI_score(outputs, targets)
            for csi_key in csi_scores:
                if csi_scores[csi_key] != 'nan':
                    if csi_key in csi_scores_train:
                        csi_scores_train[csi_key] += csi_scores[csi_key]
                        csi_seen_train[csi_key] += 1
                    else:
                        csi_scores_train[csi_key] = csi_scores[csi_key]
                        csi_seen_train[csi_key] = 1

            loop.set_postfix({'Train Loss' : training_loss})

        csi_scores_train = normalize_dictionnary_values(csi_scores_train, csi_seen_train, 2)
        print("[Train] CSI_scores : ", csi_scores_train)

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

            csi_scores = CSI_score(outputs, targets)
            for csi_key in csi_scores:
                if csi_scores[csi_key] != 'nan':
                    if csi_key in csi_scores_val:
                        csi_scores_val[csi_key] += csi_scores[csi_key]
                        csi_seen_val[csi_key] += 1
                    else:
                        csi_scores_val[csi_key] = csi_scores[csi_key]
                        csi_seen_val[csi_key] = 1

        csi_scores_val = normalize_dictionnary_values(csi_scores_val, csi_seen_val, 3)

        print(f"[Validation] Loss : {validation_loss:.2f}")
        print("[Validation] CSI_scores", csi_scores_val)
        print("\n")

        writer.add_scalar('Loss/train', training_loss, epoch)
        writer.add_scalar('Loss/test', validation_loss, epoch)

        for csi_key in csi_scores_train:
            writer.add_scalar(csi_key + '/train', csi_scores_train[csi_key], epoch)
            writer.add_scalar(csi_key + '/valid', csi_scores_val[csi_key], epoch)


    save_pred_images(network, val, n_plots=30, output_dir='./images/', device=device)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    network = cnn_2D(input_length=12, output_length=5, filter_number=70)
    summary(network, input_size=(12, 128, 128), device='cpu')
    network.to(device=device)
    train_network(network, input_length=12, output_length=5, epochs=5, batch_size=8, device=device)
