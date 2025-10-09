import torch

def convention_stretch(t):
    t = t*3
    t = t.clamp(0,1)
    return t