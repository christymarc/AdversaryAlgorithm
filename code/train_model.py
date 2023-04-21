from fastai.text.all import *

from argparse import ArgumentParser
import os

import torch
torch.cuda.set_device(2)

def train_model(name, datasize, max_size):
    
    path = f"../data/imdb{datasize}"
    print(path)
    
    model_name = f"model{name}"
    
    # if model already exists, don't override
    model_file = os.path.join("./models", model_name + ".pth")
    if (os.path.exists(model_file)):
        return
    
    
    # creating datablocks
    
    dls = DataBlock(
        blocks=(TextBlock.from_folder(path, max_vocab=max_size),CategoryBlock),
        get_y = parent_label,
        get_items=partial(get_text_files, folders=['train', 'test']),
        splitter=GrandparentSplitter(valid_name='test')
    )
    
    dls = dls.dataloaders(path)
    
    # training
    
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy, cbs=[CSVLogger(append=True)])

    print(f"{model_name} has begun training")
    
    learn.fit_one_cycle(1, 1e-2)
    
    learn.freeze_to(-2)
    learn.fit_one_cycle(6, 1e-4)
    
    learn.freeze_to(-3)
    learn.fit_one_cycle(6, 1e-4)
    
    learn.unfreeze()
    learn.fit_one_cycle(10, 1e-4)
    
    # saving model and training log
    
    learn.save(model_name)
    
    default_name = "history.csv"
    log_name = f"log{name}.csv"
    log_path = "../logs"
    
    os.rename(default_name, log_name)
    shutil.move(log_name, log_path)
    
    print(f"{model_name} trained")

def main():
    argparser = ArgumentParser("Running training")
    argparser.add_argument("name", type=str, help="extension of name for saved model and log data")
    argparser.add_argument("datasize", type=int, help="which datasize it is training")
    argparser.add_argument("max_size", type=int, help="max size of vocab")
    args = argparser.parse_args()
    
    train_model(args.name, args.datasize, args.max_size)
    
if __name__ == "__main__":
    main()
    
    
    
 