import torch
# /data3/SEN2NEON


# 1. Get Data
from data.dataset import SEN2NEON
from data.datamodule import SEN2NEONDataModule

datamodule = SEN2NEONDataModule(
    lr_dir="/data3/SEN2NEON/processed/neon_10m_linearized",
    hr_dir="/data3/SEN2NEON/processed/neon_2.5m_linearized",
    pattern="*.tif",
    crop_size_lr=128,
    dtype=torch.float32,
    allow_nan=True,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
datamodule.setup(stage="predict") # set up datamodule for prediction
loader = datamodule.predict_dataloader() # get the prediction dataloader
batch = next(iter(loader)) # get a batch from the dataloader
print(batch["lr"].shape, batch["hr"].shape, batch["name"]) # print shapes of lr, hr, and names in the batch

# 2. Get Evaluator
from metrics.validator import SREvaluator
evaluator = SREvaluator()

# 3. Get Model
from models.standard_sen2sr_model import get_standard_sen2sr_model
model = get_standard_sen2sr_model()
device = "cuda" if torch.cuda.is_available() else "cpu" # set device
model = model.to(device) # Move Model to Device

# 4. Run SR and Evaluate
for i, batch in enumerate(loader):
    print(f"Processing batch {i+1}/{len(loader)}: {batch['name']}")
    lr = batch["lr"].to(device) # Move LR to Device
    hr = batch["hr"].to(device) # Move HR to Device
    
    # extract bands
    lr = lr[:,:10,:,:] # Sentinel-2 bands 1-10
    hr = hr[:,:10,:,:] # Sentinel-2 bands 1-10
    
    with torch.no_grad(): # no grad for inference
        sr = model.forward(lr) # run SR
    for lr_,sr_,hr_ in zip(lr,sr,hr):
        results = evaluator.evaluate(lr_, sr_, hr_) # evaluate metrics
        print(f"Results for batch {i+1}: {results}")
