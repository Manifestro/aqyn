"""Train a model. Example:

python scripts/train.py --config configs/ljspeech_cfc.yaml
python scripts/train.py --config configs/ljspeech_cfc.yaml data.max_frames_per_batch=2000
"""

from mimicfc.train import main

if __name__ == "__main__":
    main()
