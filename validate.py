# Imports
import torch,os
from tqdm import tqdm
from utils.plotting import save_batch_visualizations
from utils.histogram_match import hist_match_to_reference


os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU device
device = "cuda" if torch.cuda.is_available() else "cpu" # set device


# 1. Get Data
from data.datamodule import SEN2NEONDataModule
datamodule = SEN2NEONDataModule(
    csv_path="/data3/SEN2NEON/metadata.csv",
    root_dir="/data3/SEN2NEON",
    split="validation",     # or None to ignore CSV split
    batch_size=1,
    crop_size_lr=128,
    )
datamodule.setup(stage="predict") # set up datamodule for prediction
loader = datamodule.predict_dataloader() # get the prediction dataloader
#example_batch = next(iter(loader)) # get an example batch

# 2. Get Evaluator and Metrics Sink
from metrics.validator import SREvaluator
evaluator = SREvaluator()
from metrics.unify_metrics import MetricsSink
sink = MetricsSink(save_dir="logs/metrics")

# 3 Model Loop
from models.model_selector import get_model
models_configs = { # defines input bands numbers from SEN2 and prediction function
        "srgan": {
            "bands": [4,5,6,8,10,11], # band indexes of 20m bands
            "output_bands": [0,1,2,3,4,5], # bands of the output that hold the SR 20m bands
            "predict": lambda model, x: model.predict_step(x)
        },
        "sen2sr": {
            "bands": [1,2,3,4,5,6,7,8,10,11], # band indexes or RGB-NIR + 20m bands
            "output_bands": [4,5,6,7,8,9], # bands of the output that hold the SR 20m bands
            "predict": lambda model, x: model.forward(x)
        },
        "lite_sen2sr": {
            "bands": [1,2,3,4,5,6,7,8,10,11],
            "output_bands": [4,5,6,7,8,9],
            "predict": lambda model, x: model.forward(x)
        },
        "ldsrs2": {
            "bands": [1,2,3,4,5,6,7,8,10,11],
            "output_bands": [4,5,6,7,8,9],
            "predict": lambda model, x: model.forward(x)
        },
    }
# Debugging flags
DEBUG = False

# Evaluation Loop
for model_name in models_configs.keys():
    # Get Settings
    bands_selection = models_configs[model_name]["bands"]
    predict_func = models_configs[model_name]["predict"]
    output_bands = models_configs[model_name]["output_bands"]
    
    # Load Model
    print(f"Loading model: {model_name}")
    model = get_model(model_name)
    model = model.to(device) # Move Model to Device

    print("Run Validation...")
    # 4. Run SR and Evaluate
    for i, batch in tqdm(enumerate(loader),desc=f"Validating {model_name}",total=len(loader)):
        
        # extract data from batch and normalize
        lr,hr,meta = batch["lr"],batch["hr"],batch["meta"]
        lr,hr = lr.float()/10000., hr.float()/10000. # normalize to 0-1
        
        # extract bands
        lr = lr[:, bands_selection, :, :].to(device) # Sentinel-2 6-20m bands
        hr = hr[:, bands_selection, :, :].to(device) # Sentinel-2 6-20m bands
        
        if model_name in ["srgan"]:
            # SRGAN outputs 20m bands only, so upsample LR to 20m for metrics
            lr = torch.nn.functional.interpolate(lr, scale_factor=0.5, mode="bilinear", align_corners=False)        
        
        # Run SR
        with torch.no_grad(): # no grad for inference
            sr = predict_func(model, lr) # run SR
        
        # from output, keep only 6 20m bands if RGB-NIR comes with it
        sr = sr[:,output_bands,:,:]
        hr = hr[:,output_bands,:,:]
        lr = lr[:,output_bands,:,:]
        
        # hist match the SR output to LR reference
        sr = hist_match_to_reference(target=sr, reference=lr)
        
        # Save Visualizations
        assert sr.shape == hr.shape, f"SR and HR shapes do not match: {sr.shape} vs {hr.shape}"      

        # Calculate Metrics only when not debugging
        if DEBUG==False:
            for lr_,sr_,hr_ in zip(lr,sr,hr):
                lr_,sr_,hr_ = lr_.cpu(), sr_.cpu(), hr_.cpu()
                metrics = evaluator.evaluate(lr_, sr_, hr_)
                sink.log_batch(model_name, i, batch["meta"], metrics)
            
        if i<=100: # save visualizations for first 100 batches  
            save_batch_visualizations(lr, sr, hr, meta=meta, model_name=model_name, out_root="visualizations/inf4")

        # Debugging: limit number of batches
        if DEBUG==True:
            if i==4:
                break
        
        # to be sure, flush metrics more often
        if i%100==0 and i>0 and DEBUG==False:
            out = sink.flush()
           
    # Flush after each Model
    if DEBUG==False:
        out = sink.flush()  # writes logs/metrics/val_metrics.csv
