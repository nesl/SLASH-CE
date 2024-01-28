'''
Codes are adapted from https://github.com/ML-KULeuven/deepproblog/blob/master/src/deepproblog/examples/MNIST/data/__init__.py
'''

import itertools
import json
import random
from pathlib import Path
from typing import Callable, List, Iterable, Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from fsm import Event0, Event1



transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str('./data/'), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str('./data/'), train=False, download=True, transform=transform
    ),
}


class MNIST_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]


MNIST_train = MNIST_Images("train")
MNIST_test = MNIST_Images("test")


# Define customize function here

n_event_class = 3
fsm_list = [Event0(), Event1()]

def complex_func(x: List[int]) -> int:
    """Generatr pattern labels for a 3-number MNIST sequence""" 
    for e in fsm_list:
        if e.check(x) is True: return e.label
    return n_event_class - 1


def complex_pattern(n: int, dataset: str, mnist_datasets: dict, seed=None):
    """Returns a dataset for one-digit addition"""
    return MNISTOperator(
        dataset_name=dataset,
        function_name="complex pattern",
        operator=complex_func,
        arity=n,
        seed=seed,
        datasets=mnist_datasets,
    )


class MNISTOperator(Dataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l = self.data[index]
        label = self._get_label(index)
        l = [self.dataset[i][0] for i in l]
        return l[0], l[1], l[2], label

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        arity=2,
        seed=None,
        datasets=datasets,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(MNISTOperator, self).__init__()
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.arity = arity
        self.seed = seed
        mnist_indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(mnist_indices)
        dataset_iter = iter(mnist_indices)
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
        ARGS :- ONE_DIGIT_NUMBER,...
        ONE_DIGIT_NUMBER :- mnist_img_id
        """
        if filename is None:
            filename = self.dataset_name
        file = filename
        data_text = [' '.join(str(j) for j in self.data[i]) + ' ' + str(self._get_label(i)) + '\n' for i in range(len(self))]
        with open(file, 'w') as txtfile:
            txtfile.writelines(data_text)

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    def _get_label(self, i: int):
        mnist_indices = self.data[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            self.dataset[i][1] for i in mnist_indices
        ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result

    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    seed = 0
    arity = 3

    mnist_train_data = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    mnist_test_data = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

    train_filter = (mnist_train_data.targets==0) | (mnist_train_data.targets==1) | (mnist_train_data.targets==2) | (mnist_train_data.targets==3)
    mnist_train_data.data, mnist_train_data.targets = mnist_train_data.data[train_filter], mnist_train_data.targets[train_filter]

    test_filter = (mnist_test_data.targets==0) | (mnist_test_data.targets==1) | (mnist_test_data.targets==2) | (mnist_test_data.targets==3)
    mnist_test_data.data, mnist_test_data.targets = mnist_test_data.data[test_filter], mnist_test_data.targets[test_filter]

    mnist_datasets = {
        "train": mnist_train_data,
        "test": mnist_test_data,
    }

    train_data = complex_pattern(n=arity, dataset="train", mnist_datasets=mnist_datasets, seed=seed)
    test_data = complex_pattern(n=arity, dataset="test", mnist_datasets=mnist_datasets, seed=seed)
    train_data.save_to_txt("data/labels/train_data_s"+str(arity)+".txt")