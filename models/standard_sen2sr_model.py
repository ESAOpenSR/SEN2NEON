import mlstac
import torch

def get_standard_sen2sr_model():
    # Band order: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    device = "cuda" if torch.cuda.is_available() else "cpu" # set device
    
    try: # try to load model, if not found, download it
        model = mlstac.load("model/SEN2SR/").compiled_model(device=device) # load compiled model
    except: # otherwise download it first
        mlstac.download(
            file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SR/main/mlm.json",
            output_dir="model/SEN2SR/")
        model = mlstac.load("model/SEN2SR/").compiled_model(device=device) # load compiled model
    model = model.to(device) # Move Model to Device
    return model

if __name__ == "__main__":
    model = get_standard_sen2sr_model()
    device = "cuda" if torch.cuda.is_available() else "cpu" # set device
    lr = torch.rand(1,10,128,128).to(device) # create random input
    sr = model.forward(lr) # run SR
    print(sr.shape)