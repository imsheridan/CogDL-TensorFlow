import os
import sys

sys.path.append('../')

def test_prone():

    os.system("python ../scripts/train.py --task unsupervised_node_classification --dataset wikipedia --model prone --seed 0 1 2 3 4 --hidden-size 2")
    pass

if __name__ == "__main__":
    test_prone()