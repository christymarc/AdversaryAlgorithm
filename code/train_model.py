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
    models_folder = "./models"
    model_name_ext = model_name + ".pkl"
    model_file = os.path.join(models_folder, model_name_ext)
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
    
    log_name = f"log{name}.csv"
    
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy, path=models_folder, cbs=CSVLogger(fname=log_name, append=True))

    print(f"{model_name} has begun training")
    
    learn.fit_one_cycle(1, 1e-2)
    
    learn.freeze_to(-2)
    learn.fit_one_cycle(6, 1e-4)
    
    learn.freeze_to(-3)
    learn.fit_one_cycle(6, 1e-4)
    
    learn.unfreeze()
    learn.fit_one_cycle(10, 1e-4)
    
    # saving model and training log
    
    # remove csvlogger before exporting to avoid pickling bug with fastai v2
    learn.remove_cb(CSVLogger)
    learn.export(model_name_ext)
    
    # move logs
    log_name = os.path.join(models_folder, f"log{name}.csv")
    log_path = "../logs"

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
    
    
    
 