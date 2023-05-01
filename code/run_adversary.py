from subprocess import run
import os
    
models_path = "./models"
data = "../data/examples10.csv"

for model in os.listdir(models_path):
    learner_path = os.path.join(models_path, model)
    print(learner_path)
    if ".ipynb_checkpoints" in learner_path or ".csv" in learner_path: # accounting for weird fastai thing
        continue
    if (os.path.exists(f"../data/{model}_adverse_examples10.csv")):
        continue
    run(
            [
                "python",
                "adversary.py",
                model,
                learner_path,
                data
            ]
        )
