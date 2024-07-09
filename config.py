def get_config():
    return {
        "BATCH_SIZE" : 10,
        "EPOCHS" : 28,
        "LEARNING_RATE" : 1e-4,
        "NUM_CLASSES" : 10,
        "PATCH_SIZE" : 16,
        "IMG_SIZE" : 224,
        "IN_CHANNELS" : 3,
        "NUM_HEADS" : 8,
        "DROPOUT" : 0.001,
        "HIDDEN_DIM" : 512,
        "ADAM_WEIGHT_DECAY" : 0,
        "ADAM_BETAS" : (0.9, 0.999),
        "ACTIVATION" : "gelu",
        "NUM_ENCODERS" : 4,
        "EMBED_DIM" : 64,
        "mean" : [0.485, 0.456, 0.406], 
        "std" : [0.229, 0.224, 0.225]
    }
