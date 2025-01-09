# quickstart
```bash
conda create -n data_augmentation python=3.11
conda activate data_augmentation
pip install -r requirements.txt
```

# project file structure

```plaintext
├── ./
│   └── test.ipynb
│   └── README.md
│   ├── scripts/
│   │   └── sample.py
│   │   └── parse_args.py
│   │   └── eval.py
│   │   └── utils_model.py
│   │   └── data_process.py
│   │   └── iter.py
│   │   └── find_path.py
│   │   └── response.py
│   │   └── temp.sh
│   │   └── data_loader.py
│   │   └── trainer.py
│   │   └── utils.py
│   │   └── requirements.txt
│   │   └── prompt.py
│   │   └── roberta_mlp.py
│   │   └── README.md
│   │   ├── results/
│   ├── results/
│   ├── .git/
│   ├── data/
│   │   ├── test/
│   │   ├── train/
│   ├── vast/
│   │   └── raw_train_all_onecol.csv
│   │   └── raw_val_all_onecol.csv
│   │   └── raw_test_all_onecol.csv
```

# quick build baseline
```bash
cd scripts/
bash build_baseline.sh
```

# iterative operation
```bash
cd scripts/
python iter.py # you can change setting in this file
```