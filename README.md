# Classfication-Tasks-with-Noise

## Set up Conda Environment

python --version: Python 3.10.19

pip install -r requirements.txt

## Run Code

1) Run _get_backend_noise_model.py_ to generate .pkl noise model files.

2) Run _mnist-noise-hybrid.py_ to execute code.
- parser.add_argument("--noise-train", action="store_true", default=False,
                        help="enable noise model during training")      # Set 'True' to run training with noise
- parser.add_argument("--noise-infer", action="store_true", default=True,
                        help="enable noise model during inference")     # Set 'True' to run inference with noise
