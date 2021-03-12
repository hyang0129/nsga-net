import argparse
from loguru import logger
import sys
import os


arg_string = '''
--mode client
--search_space micro 
--init_channels 32 
--n_gens 30 
--epochs 20
--layers 8 
--pop_size 40 
--n_offspring 20 
--weight_init lammarckian 
--batch_size 512
--save_path /content/drive/Shareddrives/KAGGLE/NSGANET/LAMMARCKIAN
'''


parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
parser.add_argument('--mode', type=str, default='client', help='run as either a client or a server')
parser.add_argument('--save', type=str, default='GA-BiObj', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--search_space', type=str, default='micro', help='macro or micro search space')
# arguments for micro search space
parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--n_ops', type=int, default=9, help='number of operations considered')
parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
# arguments for macro search space
parser.add_argument('--n_nodes', type=int, default=4, help='number of nodes per phases')
# hyper-parameters for algorithm
parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=50, help='population size')
parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
# arguments for back-propagation training during search
parser.add_argument('--init_channels', type=int, default=24, help='# of filters for first cell')
parser.add_argument('--layers', type=int, default=11, help='equivalent with N = 3')
parser.add_argument('--epochs', type=int, default=25, help='# of epochs to train during architecture search')
parser.add_argument('--weight_init', type=str, default="xavier", help='weight inheritance or initialization method. xavier, kaiming, or lammarckian')

parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

parser.add_argument('--save_path', type=str, default='/content/drive/Shareddrives/KAGGLE/NSGANET/LAMMARCKIAN', help='path for running and saving the experiment')

args = parser.parse_args(arg_string.split())
args.save = args.save_path

logger.add(sys.stderr,  format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", level="INFO")
logger.add(os.path.join(args.save, 'log.txt') ,format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", level="INFO", backtrace=True, diagnose=True)


