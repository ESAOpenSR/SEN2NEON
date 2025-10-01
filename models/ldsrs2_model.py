import opensr_model # import pachage
from io import StringIO
import requests
from omegaconf import OmegaConf
import torch

def get_ldsrs2_model():
    config_url = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/opensr_model/configs/config_10m.yaml"
    response = requests.get(config_url)
    config = OmegaConf.load(StringIO(response.text))

    device = "cuda" if torch.cuda.is_available() else "cpu" # set device
    model = opensr_model.SRLatentDiffusion(config, device=device) # create model
    model.load_pretrained(config.ckpt_version) # load checkpoint
    return model


if __name__ == "__main__":
    model = get_ldsrs2_model()
    lr = torch.rand(1,4,128,128).to(model.device) # create random input
    sr = model.forward(lr) # run SR
    print(sr.shape)