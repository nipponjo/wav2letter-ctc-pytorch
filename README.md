# wav2letter-ctc-pytorch


Wave2Letter ([paper](https://arxiv.org/abs/1609.03193)) with a waveform input.

The model was trained on [LibriSpeech-960](https://www.openslr.org/12/). In training, BatchNorm and Dropout were used, which can be fused into the weights to make them compatible with the [`Wave2Letter`](https://pytorch.org/audio/stable/generated/torchaudio.models.Wav2Letter.html) from `torchaudio.models`.

Pretrained weights

for `model.Wav2Letter` ([link](https://drive.google.com/u/1/uc?id=1D1Tnh0EYtUEfQq9cNtnAELn8sb3jhw9H&export=download))

for `torchaudio.models.Wav2Letter` ([link](https://drive.google.com/u/1/uc?id=12Wgf4SS3_2kubGWnXRODUgE02-ygbOAo&export=download))

Greedy decoding
|dataset|CER|WER|
|---|---|---|
|dev-clean|0.111|0.331|
|test-clean|0.105|0.318|

Example
```python
from torchaudio.models import Wav2Letter
model = Wav2Letter(num_classes=len(labels)).cuda()
model.load_state_dict(torch.load('./pretrained/states_fused.pth'))
```

Some filter kernels from the first Conv1d layer
<div align="center">
  <img src="https://user-images.githubusercontent.com/28433296/216795271-bb787682-46f7-4cd6-98c7-f19037d5d146.png" width="66%">
</div>