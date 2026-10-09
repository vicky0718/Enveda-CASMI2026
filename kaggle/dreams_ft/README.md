DreaMS fine-tuning kernel. `train_dreams.py` is generated — edit `train_src.py` / `src/casmi/dreams_backbone.py`, then

    python kaggle/bundle.py kaggle/dreams_ft/train_src.py kaggle/dreams_ft/train_dreams.py dreams_backbone
    python scripts/kaggle_api.py push kaggle/dreams_ft/kernel-metadata.json
