import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import cnn_2D
from dataset import MeteoDataset


def train_network(network, input_length, output_length, epochs, batch_size, device):

    writer = SummaryWriter()

    train = MeteoDataset(rain_dir='../Data/MeteoNet-Brest/rainmap/train', input_length=input_length,  output_length=output_length)
    val = MeteoDataset(rain_dir='../Data/MeteoNet-Brest/rainmap/val', input_length=input_length, output_length=output_length)

    n_examples_train = len(train)
    n_examples_valid = len(val)

    train_dataloader = DataLoader(train, batch_size=batch_size)
    valid_dataloader = DataLoader(val, batch_size=batch_size)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):

        network.train()
        training_loss = 0.0
        validation_loss = 0.0

        loop = tqdm(train_dataloader)
        loop.set_description(f"Epoch {epoch+1}/{epochs}")

        for batch_idx, sample in enumerate(loop):
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() / n_examples_train


            loop.set_postfix({'Train Loss' : training_loss})

        network.eval()

        for sample in valid_dataloader:
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item() / n_examples_valid

        print(f"Validation Loss : {validation_loss:.2f}")

        writer.add_scalar('Loss/train', training_loss, epochs)
        writer.add_scalar('Loss/test', validation_loss, epochs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')
network = cnn_2D(input_length=12, output_length=5, filter_number=70)
summary(network, input_size=(12, 128, 128), device='cpu')
network.to(device=device)
train_network(network, input_length=12, output_length=5, epochs=5, batch_size=8, device=device)
