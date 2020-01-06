import os
import sys

sys.path.append('../')

def test_node2vec():

    os.system("python ../scripts/train.py --task unsupervised_node_classification --dataset wikipedia --model node2vec --p_value 0.3 --q_value 0.7 --seed 0 1 2 3 4")
    pass

if __name__ == "__main__":
    test_node2vec()