import os
import sys

sys.path.append('../')

def test_deepwalk():

    os.system("python ../scripts/train.py --task unsupervised_node_classification --dataset wikipedia --model deepwalk --seed 0 1 2 3 4")
    pass

if __name__ == "__main__":
    test_deepwalk()