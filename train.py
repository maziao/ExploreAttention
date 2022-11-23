from config import Config
from data_manager import DataManager
from dataloader import Dataloader
from hybrid_model import HybridModel
from trainer import Trainer


if __name__ == '__main__':
    config = Config()
    datamanager = DataManager(config)
    dataloader = Dataloader(config, datamanager)
    network = HybridModel(config, datamanager)
    trainer = Trainer(config, network, dataloader)
    trainer.train()
    trainer.save_model()
    trainer.save_training_log()
    trainer.visualize()
