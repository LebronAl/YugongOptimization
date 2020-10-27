import configparser
from data import *
from model import *
from result import *


if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read('config.ini')
    result = Result()
    model = Model(Data(conf), result)
    model.tableMigration()
    result.persist()
    print("All Done")
