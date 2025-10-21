def get_srgan_model():
    from opensr_srgan import load_inference_model
    model = load_inference_model("SWIR")
    return model

if __name__=="__main__":
    print("Testing SRGAN model loading...")
    model = get_srgan_model()
    print("SRGAN model loaded successfully.")