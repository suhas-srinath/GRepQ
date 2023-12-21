# Configuration file for GRepQ.

exp_config = {

    'run_type': 'll_model_train',  # 'll_model_train' or 'hl_model_train'
    'database_path': str(r"../Databases"),

    # Training parameters
    'datasets': {
        # Train datasets
        'LIVE_FB_synthetic': {'train': True},
    },

    'model': None,  # Model being trained and tested
    'resume_training': False,  # Resume training from existing checkpoint
    'resume_path': str(r"./checkpoint.tar"),  # Last checkpoint path if resuming training

    'epochs': 15, 
    'lr_update': 20,  # Update learning rate after specified no. of epochs
    'test_epoch': 3,  # Validate after these many epochs of training
    'lr_decay': 1.0,

    # Low Level Model arguments
    'batch_size_qacl': 8,  # 9 frames in 1 batch
    'lr_llm': 1e-4,
    'pristine_img_dir': str(r"../Databases/Pristine"),
    'patch_size': 96,
    'device': "cuda",
    'sharpness_param': 0.75,
    'colorfulness_param': 0.8,
    'results_path_llm': str(r"./Results/LLM"),

    # High Level Model arguments
    'crop': 'center',
    'crop_size': (224, 224),
    'batch_size_gcl': 128,
    'tau': 32,  # temperature parameter
    'lr_hlm': 1e-6,
    'results_path_hlm': str(r"./Results/HLM"),
}