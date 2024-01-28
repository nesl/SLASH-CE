import sys
sys.path.append('../')
import pickle
import numpy as np 
import random
from typing import Callable, List, Iterable, Tuple
import torch
import torchvision.transforms as transforms

from fsm import Event0, Event1





def generate_prim(label_map):
    # Filter and relabel the classes
    primitive_data = np.load('./data/nursing_data/primitive_dataset.npz')
    x_prim_train, x_prim_test, y_prim_train, y_prim_test = primitive_data['x_prim_train'], primitive_data['x_prim_test'], primitive_data['y_prim_train'], primitive_data['y_prim_test']

    idx_train = []
    idx_test = []
    for old_label, new_label in label_map.items():
        idx_train.append(np.where(y_prim_train == old_label)[0])
        idx_test.append(np.where(y_prim_test == old_label)[0])
        y_prim_train[y_prim_train == old_label] = new_label
        y_prim_test[y_prim_test == old_label] = new_label

    idx_train = np.concatenate(idx_train)
    idx_test = np.concatenate(idx_test)

    x_prim_train = x_prim_train[idx_train]
    y_prim_train = y_prim_train[idx_train]
    x_prim_test = x_prim_test[idx_test]
    y_prim_test = y_prim_test[idx_test]

    with open("./data/train.pkl","wb") as f:
        pickle.dump((x_prim_train, y_prim_train),f)
    with open("./data/test.pkl","wb") as f: 
        pickle.dump((x_prim_test, y_prim_test),f)
    return {"train": (x_prim_train, y_prim_train), "test": (x_prim_test, y_prim_test)}


''' Generate complex event training/test set using prmitive activities '''

fsm_list = [Event0(), Event1()]
n_event_class = 3 

    
def complex_func(x: List[int]) -> int:
    """Generatr pattern labels for a 3-number MNIST sequence""" 
    for e in fsm_list:
        if e.check(x) is True: return e.label
    return n_event_class - 1


def complex_pattern(n: int, dataset: str, prim_datasets: dict, seed=None):
    """Returns a dataset for one-digit addition"""
    return CEGenerator(
        dataset_name=dataset,
        function_name="complex pattern",
        operator=complex_func,
        arity=n,
        seed=seed,
        datasets=prim_datasets,
    )


class CEGenerator():
    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        datasets,
        arity=3,
        seed=None,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.arity = arity
        self.seed = seed
        prim_indices = list(range(len(self.dataset[0])))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(prim_indices)
        dataset_iter = iter(prim_indices)
        # Build list of examples (mnist indices)
        self.data = []
        try:
            while dataset_iter:
                self.data.append(
                    [
                        next(dataset_iter) for _ in range(self.arity)
                    ]
                )
        except StopIteration:
            pass

    def save_to_txt(self, filename=None):
        """
        Save to a TXT file (for one digit usage).

        Format is (EXAMPLE) for each line
        EXAMPLE :- ARGS,expected_result
        ARGS :- ONE_PRIM_DATA,...
        ONE_PRIM_DATA :- prim_data_id
        """
        data_text = [' '.join(str(j) for j in self.data[i]) + ' ' + str(self._get_label(i)) + '\n' for i in range(len(self))]
        # print(data_text)
        with open(filename, 'w') as txtfile:
            txtfile.writelines(data_text)

    def _get_label(self, i: int):
        prim_indices = self.data[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            self.dataset[1][i] for i in prim_indices
        ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    seed = 0
    label_map = {4:0, 2:1, 12:2, 9:3}
    prim_datasets = generate_prim(label_map)
    arity = 3
    train_data = complex_pattern(n=arity, dataset="train", prim_datasets=prim_datasets, seed=seed)
    train_data.save_to_txt("data/labels/train_data_s"+str(arity)+".txt")
    test_data = complex_pattern(n=arity, dataset="test", prim_datasets=prim_datasets, seed=seed)
    test_data.save_to_txt("data/labels/test_data_s"+str(arity)+".txt")
    # file = open("./data/train.pkl",'rb')
    x_prim_train, y_prim_train = np.load("./data/train.pkl",allow_pickle=True)
    print(x_prim_train.shape,y_prim_train.shape)
