from data.robotic_dataloader import get_train_dataloader, get_val_dataloader
from utils.options import Options
from data.utils.prepare_data import get_split

from train_engine_contour import TrainEngine

if __name__ == '__main__':
    opt = Options().opt
    train_files, test_files = get_split(opt.fold)
    train_dataloader = get_train_dataloader(train_files, opt)
    val_dataloader = get_val_dataloader(test_files, opt)
    engine = TrainEngine(opt)
    engine.set_data(train_dataloader, val_dataloader)
    engine.train_model()

