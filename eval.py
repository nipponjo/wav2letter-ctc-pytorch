# %%
import torch
import torch.nn as nn
import torchaudio
from labels import labels
from torch.utils.data import DataLoader
# from model import Wav2Letter as Wav2Letter_
from torchaudio.models import Wav2Letter
from utils import GreedyLM, collate_fun, test

# %%

batch_size = 16
lm = GreedyLM(labels)

testset_url = 'dev-clean'
#testset_url = 'test-clean'
test_dataset = torchaudio.datasets.LIBRISPEECH('G:/data', url=testset_url)

test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size, shuffle=False,
                              collate_fn=lambda x: collate_fun(x, lm.encode, 'valid'))

criterion = nn.CTCLoss(blank=len(labels)-1, zero_infinity=True).cuda()

model = Wav2Letter(num_classes=len(labels)).cuda()
model.load_state_dict(torch.load('./pretrained/states_fused.pth'))
model.eval()

# %%

test_loss, cer, wer = test(model, test_loader, criterion, lm)
print(f"Test loss: {test_loss}")
print(f"CER: {cer}")
print(f"WER: {wer}")
