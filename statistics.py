from data_manager import DataManager
from config import Config


class Statistics(Config):
    def __init__(self):
        super().__init__()
        self.num_category = 19
        self.remove_entity_mark = True
        self.datamanager = DataManager(self)


if __name__ == '__main__':
    statistics = Statistics()
