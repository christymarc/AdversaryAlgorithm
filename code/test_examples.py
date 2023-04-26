from fastai.text.all import *

import torch
torch.cuda.set_device(2)

def narrow_down(pos_path, neg_path, models_path):
    text = []
    category = []
    sentiment = 'pos'
    count = 0
    learners = []
    for model in os.listdir(models_path):
        learner_path = os.path.join(models_path, model)
        if learner_path == "models/.ipynb_checkpoints" or ".csv" in learner_path: # accounting for weird fastai thing
            continue
        l = load_learner(learner_path, cpu=False)
        learners.append(l)
    for path in [pos_path, neg_path]:
        if path == neg_path:
            sentiment = 'neg'
            count = 0
        for filename in os.listdir(path):
            f = os.path.join(path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                with open(f, 'rt') as fd:
                    line = fd.readlines()[0] # ensure text isn't added to text as a list
                    i = 0
                    correct = 0
                    for learner in learners:
                        if (learner.predict(line)[0] == sentiment): 
                            correct += 1
                        i += 1
                    if (i == correct):
                        text.append(line)
                        if (sentiment == 'pos'): # either 0 for negative or 1 for positive
                            category.append(1)
                        else:
                            category.append(1)
                        count += 1
                        print("added")
                        if (count >= 500):
                            break
    
    d = {"text":text, "category":category}
    df = pd.DataFrame(data=d)
    
    pathfile = "../data/examples1000.csv"
    df.to_csv(pathfile, index=False)
          
        
def main():
    pos_path = "../data/imdb10000/test/pos"
    neg_path = "../data/imdb10000/test/neg"
    models_path = "models"
    narrow_down(pos_path, neg_path, models_path)

    
if __name__ == "__main__":
    main()