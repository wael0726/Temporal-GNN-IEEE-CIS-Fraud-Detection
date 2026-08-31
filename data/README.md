# IEEE-CIS data

The final project bundle includes the original `ieee-fraud-detection.zip` archive here so the portfolio ZIP stays practical to transfer.

Extract it with:

```bash
python scripts/prepare_data.py --data-dir data
```

This creates:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

Then validate with:

```bash
python scripts/validate_dataset.py --data-dir data
```

The raw CSV files are ignored by Git and should not be committed to a public portfolio repository.
