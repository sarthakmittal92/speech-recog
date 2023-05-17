# PyTorch things
import pickle
from torchaudio.datasets import SPEECHCOMMANDS
import torch
import torchaudio
import torch.nn.functional as F

# Other libs
from urllib.request import urlopen
import matplotlib.pyplot as plt
import glob
import os
import random
from tqdm.notebook import tqdm
import torchsummary
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import pandas as pd
import seaborn as sn

# basic random seed
import os
import random
import numpy as np
# tensorflow random seed
import tensorflow as tf

DEFAULT_RANDOM_SEED = 2021


def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def seedTF(seed=DEFAULT_RANDOM_SEED):
    tf.random.set_seed(seed)

# torch random seed


def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# basic + tensorflow + torch


def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTF(seed)
    seedTorch(seed)


seedEverything(1004)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# with open('/content/blind_test.pkl', 'rb') as f:
#   blind = pickle.load(f)
# blind_set = [test_set[i] for i in blind]
# test_set = blind_set


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return sorted([os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj])

        if subset == "testing":
            self._walker = load_list("testing_list.txt")
            # test_list = [int(line) for line in urlopen('https://www.cse.iitb.ac.in/~pjyothi/cs753/test_list.txt')]
            with open('/content/blind_test.pkl', 'rb') as f:
                blind = pickle.load(f)
            self._walker = ['./' + self._walker[i] for i in blind]
        elif subset == "training":
            excludes = load_list("validation_list.txt") + \
                load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w[2:] not in excludes]
            train_list = [int(line) for line in urlopen(
                'https://www.cse.iitb.ac.in/~pjyothi/cs753/train_list.txt')]
            self._walker = [self._walker[i] for i in train_list]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

classes = sorted(os.listdir('./SpeechCommands/speech_commands_v0.02'))
classes.remove("LICENSE")
classes.remove("README.md")
classes.remove("_background_noise_")
classes.remove("testing_list.txt")
classes.remove("validation_list.txt")
classes.remove('.DS_Store')

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
# labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
mfcc = torchaudio.transforms.MFCC(n_mfcc=12, log_mels=True)(waveform)
print(mfcc.shape)


class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, classes, file_list):

        self.classes = classes

        # create a map from class name to integer
        self.class_to_int = dict(zip(classes, range(len(classes))))

        # store the file names
        self.samples = file_list

        # store our MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            n_mfcc=12, log_mels=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        with torch.no_grad():
            # load a normalized waveform
            waveform, sample_rate, label, speaker_id, utterance_number = self.samples[i]

            # if the waveform is too short (less than 1 second) we pad it with zeroes
            if waveform.shape[1] < 16000:
                waveform = F.pad(input=waveform, pad=(
                    0, 16000 - waveform.shape[1]), mode='constant', value=0)

            # then, we apply the transform
            mfcc = self.mfcc_transform(waveform).squeeze(0).transpose(0, 1)

        # return the mfcc coefficient with the sample label
        return mfcc, self.class_to_int[label]


if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

batch_size = 256
train_dataset = SpeechDataset(classes, train_set)
train_dl = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

test_dataset = SpeechDataset(classes, test_set)
test_dl = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)


class SpeechLSTM(torch.nn.Module):

    def __init__(self):
        super(SpeechLSTM, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=12, num_layers=2, hidden_size=350, batch_first=True, dropout=0.2
        )

        self.out_layer = torch.nn.Linear(350, 35)

    def forward(self, x):

        out, _ = self.lstm(x)

        x = self.out_layer(out[:, -1, :])

        return F.log_softmax(x, dim=1)


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_dl):

        model.zero_grad()

        data = data.to(device)
        target = target.to(device)
        # print(data.shape)

        output = model(data)
        # print(output.argmax(dim=-1))
        # print(target)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dl.dataset)} ({100. * batch_idx / len(train_dl):.2f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_dl:

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_dl.dataset)} ({100. * correct / len(test_dl.dataset):.2f}%)\n")


model = SpeechLSTM()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

log_interval = 20
n_epoch = 20

pbar_update = 1 / (len(train_dl) + len(test_dl))
losses = []

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

model.eval()
correct = 0
for data, target in test_dl:

    data = data.to(device)
    target = target.to(device)

    output = model(data)

    pred = get_likely_index(output)
    correct += number_of_correct(pred, target)

    # update progress bar
    pbar.update(pbar_update)

print(f"{100. * correct / len(test_dl.dataset):.2f}")
