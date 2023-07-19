import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import os
from logging import getLogger

import torch.cuda
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color
from CGCL import CGCL
from trainer import CGCLTrainer


device = torch.device("cuda:0")


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def run_single_model(args):
    # configurations initialization
    config = Config(
        model=CGCL,
        dataset=args.dataset,
        config_file_list=args.config_file_list
    )
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = CGCL(config, train_data.dataset).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = CGCLTrainer(config, model)

    # model training
    config['show_progress'] = False
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Netflix',
                        help='The datasets can be: yelp,amazon-books,gowalla-merged.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/CGCL.yaml'
    ]
    if args.dataset in ['yelp', 'amazon-books', 'gowalla-merged','ml-1m','Netflix', 'yelp','Epinions','book-crossing','yahoo-music', 'amazon-books', 'gowalla-merged', 'Alibaba-iFashion','douban','Douban','DoubanBook','DoubanMovie','ml-20m' ]:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config is not '':
        args.config_file_list.append(args.config)

    #parser.add_argument('--hyper_layers', type=int, default=0, help='External config file name.')
    run_single_model(args)
