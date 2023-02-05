# %%
import torch
from torchaudio.models.wav2letter import Wav2Letter
from model import Wav2Letter as Wav2Letter_
from labels import labels

# %%

model_ta = Wav2Letter(num_classes=len(labels))
model_mod = Wav2Letter_(num_classes=len(labels))

sd_mod_path = "./pretrained/states_bn_.pth"

ckpt_states = torch.load(sd_mod_path)
model_mod.load_state_dict(ckpt_states['model'])

# %%

model_mod_sd = model_mod.state_dict()
model_ta_sd = dict(model_ta.state_dict())


# %%

model_ta_sd["acoustic_model.0.0.weight"].copy_(
    model_mod_sd["acoustic_model.0.conv.weight"]
)

model_ta_sd["acoustic_model.0.0.bias"].copy_(
    model_mod_sd["acoustic_model.0.conv.bias"]
)

for i in range(0, 10):
    eps = model_mod.acoustic_model[1][i].bn.eps    
    mod_w = model_mod_sd[f'acoustic_model.1.{i}.conv.weight']
  
    mod_bn_var = model_mod_sd[f'acoustic_model.1.{i}.bn.running_var']
    mod_bn_mean = model_mod_sd[f'acoustic_model.1.{i}.bn.running_mean']
    mod_bn_w = model_mod_sd[f'acoustic_model.1.{i}.bn.weight']
    mod_bn_b = model_mod_sd[f'acoustic_model.1.{i}.bn.bias']
    model_ta_sd[f"acoustic_model.1.{2*i}.weight"].copy_(
        mod_w*mod_bn_w[:,None,None] / mod_bn_var[:,None,None].add(eps).sqrt())
    
    model_ta_sd[f"acoustic_model.1.{2*i}.bias"].copy_(
        mod_bn_b - mod_bn_mean*mod_bn_w/mod_bn_var.add(eps).sqrt()
        )


i = 10
mod_w = model_mod_sd[f'acoustic_model.1.{i}.conv.weight']
mod_b = model_mod_sd[f'acoustic_model.1.{i}.conv.bias']

model_ta_sd[f"acoustic_model.1.{2*i}.weight"].copy_(mod_w) 
model_ta_sd[f"acoustic_model.1.{2*i}.bias"].copy_(mod_b)

model_ta.load_state_dict(model_ta_sd)

# %%
model_mod.eval()
model_ta.eval()

x = torch.randn(1, 1, 16000)

out_mod = model_mod(x)
out_ta = model_ta(x)

print((out_mod - out_ta).abs().mean())

# %%

torch.save({'model': model_ta.state_dict()}, f"{sd_mod_path.removesuffix('.pth')}_fused.pth")
