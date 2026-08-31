"""Run the same small real-data experiment used for the included benchmark."""
from src.train import run

if __name__ == "__main__":
    result = run(data_dir="data", dataset="ieee_cis", max_rows=100000, epochs=8, history_k=3, tune=False)
    print(result)
