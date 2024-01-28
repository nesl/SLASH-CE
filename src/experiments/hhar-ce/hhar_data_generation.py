import sys
sys.path.append('../')
import pickle
import numpy as np 
import random
from typing import Callable, List, Iterable, Tuple
import torch
import torchvision.transforms as transforms

from fsm import Event0, Event1


hhar_data, hhar_label = np.load('data/hhar_data/data_20_120.npy'), np.load('data/hhar_data/label_20_120.npy')
hhar_label = hhar_label[:,:,2]
# Preprocess
# Filter out data when there are more than two activities take place in one data sample
filter = np.argwhere(hhar_label.min(axis=-1) == hhar_label.max(axis=-1)).squeeze(axis=-1)

hhar_data = hhar_data[filter]
hhar_data = hhar_data.transpose(0,2,1)
hhar_label = hhar_label[filter]
hhar_label = hhar_label.min(axis=-1)


def generate_prim(labels):
    class_dict = {0: "Biking", 1: "Sitting", 2: "Standing", 3: "Walking", 4: "Stair Up", 5: "Stair down"}
    prim_class_names = [class_dict[l] for l in labels]

    mask = np.argwhere([True if l in labels else False for l in hhar_label]).squeeze(axis=-1)
    x_primitive = hhar_data[mask]
    y_primitive = hhar_label[mask]
    # Relabel the classes
    y_primitive = np.array([labels.index(l) for l in y_primitive])
    dataset = split_data(x_primitive, y_primitive)
    x_prim_train, y_prim_train = dataset["train"]
    x_prim_test, y_prim_test = dataset["test"]
    # print(dataset["train"][0][1].shape,dataset["train"][1][1].shape)
    with open("./data/train.pkl","wb") as f:
        pickle.dump(dataset["train"],f)
    with open("./data/test.pkl","wb") as f: 
        pickle.dump(dataset["test"],f)
    # np.save('./data/train.npy', dataset["train"])
    # np.save('./data/test.npy', dataset["test"])
    return dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_data(data, label):
    n = len(data)
    n_train = int(n * 0.8)

    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    train_data = data[train_idx], label[train_idx]
    test_data = data[test_idx], label[test_idx]
    return {"train":train_data, "test":test_data}


''' Generate complex event training/test set using prmitive activities '''

fsm_list = [Event0(), Event1()]
n_event_class = 3 


def generate_complex_event(x_data, y_label, n_event_data):
    imu_indices = list(range(len(x_data)))
    # rng = random.Random(0)
    # rng.shuffle(imu_indices)

    dataset_iter = iter(imu_indices)
    arity = 3
    
    data = []
    try:
        while dataset_iter:
            data.append(
                [
                    next(dataset_iter) for _ in range(arity)
                ]
            )
    except StopIteration:
        pass


    
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
    set_seed(seed)
    class_dict = {0: "Biking", 1: "Sitting", 2: "Standing", 3: "Walking", 4: "Stair Up", 5: "Stair down"}
    labels = [0, 1, 3, 4]
    prim_datasets = generate_prim(labels)
    arity = 3
    train_data = complex_pattern(n=arity, dataset="train", prim_datasets=prim_datasets, seed=seed)
    train_data.save_to_txt("data/labels/train_data_s"+str(arity)+".txt")
    test_data = complex_pattern(n=arity, dataset="test", prim_datasets=prim_datasets, seed=seed)
    test_data.save_to_txt("data/labels/test_data_s"+str(arity)+".txt")
    # file = open("./data/train.pkl",'rb')
    x_prim_train, y_prim_train = np.load("./data/train.pkl",allow_pickle=True)
    print(x_prim_train[4224])
