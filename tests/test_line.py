import os
import sys

sys.path.append('../')

def test_line():

    os.system("python ../scripts/train.py --task unsupervised_node_classification --dataset wikipedia --model line --seed 0 1 2 3 4")
    pass

if __name__ == "__main__":
    test_line()