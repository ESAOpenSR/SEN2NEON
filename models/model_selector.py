def get_model(model_name):
    if model_name == "srgan":
        from models.srgan_model import get_srgan_model
        return get_srgan_model()
    elif model_name == "sen2sr":
        from models.standard_sen2sr_model import get_standard_sen2sr_model
        return get_standard_sen2sr_model()
    elif model_name == "lite_sen2sr":
        from models.lite_sen2sr_model import get_lite_sen2sr_model
        return get_lite_sen2sr_model()
    elif model_name == "ldsrs2":
        from models.ldsrs2_sen2sr_model import get_ldsrs2_sen2sr_model
        return get_ldsrs2_sen2sr_model()
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    

if __name__ == "__main__":
    models = ["srgan", "sen2sr", "lite_sen2sr", "ldsrs2"]
    for model_name in models:
        print(f"Loading model: {model_name}")
        model = get_model(model_name)
