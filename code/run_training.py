from subprocess import run

# these numbers reflect total dataset sizes of 10000, 20000, 30000, 40000, and 50000 respectively
dataset_sizes = ["2500", "5000", "7500", "10000", "12500"]
vocab_sizes = ["40000", "60000", "80000"]

for d in dataset_sizes:
    for v in vocab_sizes:
        for i in range(5):
            name = f"{d}_{v}_{i}"
            
            run(
                [
                    "python",
                    "train_model.py",
                    name,
                    d,
                    v
                ]
            )