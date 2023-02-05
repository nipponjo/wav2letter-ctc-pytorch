# %%
import os
import torch
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader
from model import Wav2Letter
from labels import labels
from utils import *
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

# %%


class ConfigObj:
    def __init__(self, config_dict):
        self.__dict__ = config_dict


config_dict = {
    'checkpoint_dir': './checkpoints/wav2letter0',
    'tblog_dir': './logs/w2l_0',
    'resume_training': '',
    'batch_size': 16,

    'lr': 3e-4,
    'epochs': 100,

}
config = ConfigObj(config_dict)

# %%

torch.manual_seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lm = GreedyLM(labels)

train_dataset100 = torchaudio.datasets.LIBRISPEECH(
    'G:/data', url='train-clean-100')
train_dataset360 = torchaudio.datasets.LIBRISPEECH(
    'G:/data', url='train-clean-360')
train_dataset500 = torchaudio.datasets.LIBRISPEECH(
    'G:/data', url='train-other-500')

train_dataset = ConcatDataset(
    [train_dataset100, train_dataset360, train_dataset500])
#train_dataset = torchaudio.datasets.LIBRISPEECH('C:/data', url='train-clean-100')

test_dataset = torchaudio.datasets.LIBRISPEECH('G:/data', url='test-clean')

kwargs = {'num_workers': 0, 'pin_memory': True}
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.batch_size, shuffle=True,
                          collate_fn=lambda x: collate_fun(
                              x, lm.encode, 'train'),
                          **kwargs)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=config.batch_size, shuffle=False,
                         collate_fn=lambda x: collate_fun(
                             x, lm.encode, 'valid'),
                         **kwargs)


# %%

if not os.path.exists(config.checkpoint_dir):
    os.makedirs(config.checkpoint_dir)
    print(f"Created checkpoint folder @ {config.checkpoint_dir}")


model = Wav2Letter(num_classes=len(labels)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), config.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.3, patience=5)


last_epoch, n_iter = 0, 0
if config.resume_training != '':
    ckpt = torch.load(config.resume_training)
    model.load_state_dict(ckpt['model'])
    if 'optim' in ckpt:
        optimizer.load_state_dict(ckpt['optim'])
    if 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    if 'epoch' in ckpt:
        last_epoch = ckpt['epoch']
    if 'iter' in ckpt:
        n_iter = ckpt['iter']


criterion = nn.CTCLoss(blank=len(labels)-1, zero_infinity=True).to(device)

# %%

writer = SummaryWriter(config.tblog_dir)


# %%

model.train()

for epoch in range(last_epoch, config.epochs):
    for batch in train_loader:

        waves, labels, input_lens, output_lens = batch
        waves, labels = waves.cuda(
            non_blocking=True), labels.cuda(non_blocking=True)

        out = model(waves)  # (batch, n_class, time)
        out = out.permute(2, 0, 1)  # (time, batch, n_class)

        optimizer.zero_grad()
        loss = criterion(out, labels, input_lens, output_lens)
        loss.backward()
        #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        grad_norm = get_norm(model.parameters())
        optimizer.step()

        writer.add_scalar('loss', loss.item(), n_iter)
        writer.add_scalar('grad_norm', grad_norm, n_iter)

        if n_iter % 100 == 0:
            print(f"loss: {loss.item()} lr: {optimizer.param_groups[0]['lr']}")
            save_checkpoint(model, optimizer, scheduler, epoch, n_iter, config)

        if n_iter % 5_000 == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value.data.cpu().numpy(), n_iter)

        if n_iter % 10_000 == 0 and n_iter > 0:
            save_checkpoint(model, optimizer, scheduler,
                            epoch, n_iter, config, backup=True)
            model.eval()
            test_loss = test_and_log(
                model, test_loader, criterion, writer, lm, n_iter)
            scheduler.step(test_loss)
            model.train()

        n_iter += 1

    test_and_log(model, test_loader, criterion, writer, n_iter)

# %%
