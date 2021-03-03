'''
Author: yuan
Date: 2021-02-23 14:14:20
LastEditTime: 2021-02-26 10:31:13
FilePath: /aidc-algorithm/image-classification/utils/loader.py
'''
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from PIL import Image


def get_training_dataloader(root_path='datasets/train', batch_size=16,
                            num_workers=2,
                            shuffle=True, size=(128, 128)):
    """ return training dataloader
    Args:
        root_path: path to training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: training_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    training_dataset = torchvision.datasets.ImageFolder(
        root=root_path, transform=transform_train)
    print(training_dataset.class_to_idx)
    training_loader = DataLoader(
        training_dataset, shuffle=shuffle,
        num_workers=num_workers, batch_size=batch_size)

    return training_loader


def get_valid_dataloader(root_path='datasets/val',
                         batch_size=16, num_workers=2,
                         shuffle=True, size=(128, 128)):
    """ return validation dataloader
    Args:
        root_path: path to validation dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: valid_loader:torch dataloader object
    """

    transform_valid = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    valid_dataset = torchvision.datasets.ImageFolder(
        root=root_path, transform=transform_valid)
    print(dir(valid_dataset))
    # print(valid_dataset[0][0].size()) # 3 ,128 , 128
    valid_loader = DataLoader(
        valid_dataset, shuffle=shuffle,
        num_workers=num_workers, batch_size=batch_size)

    return valid_loader


def get_test_dataloader(root_path, ref_path,
                        batch_size, num_workers,
                        shuffle=False, size=(128, 128),
                        ):

    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    test_dataset = CustomDataSet(
        root_path,
        ref_path,
        transform=test_transform)
    # print(test_dataset[0].size()) # 3, 128, 128
    test_loader = DataLoader(
        test_dataset, shuffle=shuffle,
        num_workers=num_workers, batch_size=batch_size)

    return test_loader


class CustomDataSet(Dataset):
    def __init__(self, root_path, ref_path, transform):
        self.root_path = root_path
        self.ref_path = ref_path
        self.transform = transform

        all_imgs = os.listdir(root_path)
        # self.total_imgs = natsort.natsorted(all_imgs)
        self.total_imgs = sorted(all_imgs)

        all_refs = os.listdir(ref_path)
        self.total_refs = sorted(all_refs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root_path, self.total_imgs[idx])
        ref_loc = os.path.join(self.ref_path, self.total_refs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        # print('img_loc, ref_loc',img_loc, ref_loc)
        return tensor_image, img_loc, ref_loc


# IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
#                   '.pgm', '.tif', '.tiff', '.webp')


# def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
#     """Checks if a file is an allowed extension.
#     Args:
#         filename (string): path to a file
#         extensions (tuple of strings): extensions to consider (lowercase)
#     Returns:
#         bool: True if the filename ends with one of given extensions
#     """
#     return filename.lower().endswith(extensions)


# def make_dataset(
#     directory: str,
#     class_to_idx: Dict[str, int],
#     extensions: Optional[Tuple[str, ...]] = None,
#     is_valid_file: Optional[Callable[[str], bool]] = None,
# ) -> List[Tuple[str, int]]:
#     """Generates a list of samples of a form (path_to_sample, class).
#     Args:
#         directory (str): root dataset directory
#         class_to_idx (Dict[str, int]): dictionary mapping class name to class index
#         extensions (optional): A list of allowed extensions.
#             Either extensions or is_valid_file should be passed. Defaults to None.
#         is_valid_file (optional): A function that takes path of a file
#             and checks if the file is a valid file
#             (used to check of corrupt files) both extensions and
#             is_valid_file should not be passed. Defaults to None.
#     Raises:
#         ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
#     Returns:
#         List[Tuple[str, int]]: samples of a form (path_to_sample, class)
#     """
#     instances = []

#     directory = os.path.expanduser(directory)
#     both_none = extensions is None and is_valid_file is None
#     both_something = extensions is not None and is_valid_file is not None
#     if both_none or both_something:
#         raise ValueError(
#             "Both extensions and is_valid_file cannot be None or not None at the same time")
#     if extensions is not None:
#         def is_valid_file(x: str) -> bool:
#             return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
#     is_valid_file = cast(Callable[[str], bool], is_valid_file)

#     for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
#         for fname in sorted(fnames):
#             path = os.path.join(root, fname)
#             if is_valid_file(path):
#                 instances.append(path)
#     # for target_class in sorted(class_to_idx.keys()):
#     #     class_index = class_to_idx[target_class]
#     #     target_dir = os.path.join(directory, target_class)
#     #     if not os.path.isdir(target_dir):
#     #         continue
#     #     for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
#     #         for fname in sorted(fnames):
#     #             path = os.path.join(root, fname)
#     #             if is_valid_file(path):
#     #                 item = path, class_index
#     #                 instances.append(item)
#     return instances


# def pil_loader(path: str) -> Image.Image:
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')


# def default_loader(path: str) -> Any:
#     return pil_loader()


# class DatasetFolder(VisionDataset):
#     """A generic data loader where the samples are arranged in this way: ::
#         root/class_x/xxx.ext
#         root/class_x/xxy.ext
#         root/class_x/[...]/xxz.ext
#         root/class_y/123.ext
#         root/class_y/nsdf3.ext
#         root/class_y/[...]/asd932_.ext
#     Args:
#         root (string): Root directory path.
#         loader (callable): A function to load a sample given its path.
#         extensions (tuple[string]): A list of allowed extensions.
#             both extensions and is_valid_file should not be passed.
#         transform (callable, optional): A function/transform that takes in
#             a sample and returns a transformed version.
#             E.g, ``transforms.RandomCrop`` for images.
#         target_transform (callable, optional): A function/transform that takes
#             in the target and transforms it.
#         is_valid_file (callable, optional): A function that takes path of a file
#             and check if the file is a valid file (used to check of corrupt files)
#             both extensions and is_valid_file should not be passed.
#      Attributes:
#         classes (list): List of the class names sorted alphabetically.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         samples (list): List of (sample path, class_index) tuples
#         targets (list): The class_index value for each image in the dataset
#     """

#     def __init__(
#             self,
#             root: str,
#             class_to_idx: Dict[str, int],
#             classes: List[str],
#             loader: Callable[[str], Any] = default_loader,
#             extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             is_valid_file: Optional[Callable[[str], bool]] = None,
#     ) -> None:
#         super(DatasetFolder, self).__init__(root, transform=transform,
#                                             target_transform=target_transform)
#         # classes, class_to_idx = self._find_classes(self.root)
#         # print('classes, class_to_idx',classes, class_to_idx)
#         samples = self.make_dataset(
#             self.root, class_to_idx, extensions, is_valid_file)

#         if len(samples) == 0:
#             msg = "Found 0 files in subfolders of: {}\n".format(self.root)
#             if extensions is not None:
#                 msg += "Supported extensions are: {}".format(
#                     ",".join(extensions))
#             raise RuntimeError(msg)

#         self.loader = loader
#         self.extensions = extensions

#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.samples = samples
#         self.targets = [s[1] for s in samples]

#     @ staticmethod
#     def make_dataset(
#         directory: str,
#         class_to_idx: Dict[str, int],
#         extensions: Optional[Tuple[str, ...]] = None,
#         is_valid_file: Optional[Callable[[str], bool]] = None,
#     ) -> List[Tuple[str, int]]:
#         return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

#     def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
#         """
#         Finds the class folders in a dataset.
#         Args:
#             dir (string): Root directory path.
#         Returns:
#             tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
#         Ensures:
#             No class is a subdirectory of another.
#         """
#         classes = [d.name for d in os.scandir(dir) if d.is_dir()]
#         classes.sort()
#         class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#         return classes, class_to_idx

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return sample, target

#     def __len__(self) -> int:
#         return len(self.samples)


# def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
#     """ return training dataloader
#     Args:
#         mean: mean of cifar100 training dataset
#         std: std of cifar100 training dataset
#         path: path to cifar100 training python dataset
#         batch_size: dataloader batchsize
#         num_workers: dataloader num_works
#         shuffle: whether to shuffle
#     Returns: train_data_loader:torch dataloader object
#     """

#     transform_train = transforms.Compose([
#         #transforms.ToPILImage(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#     #cifar100_training = CIFAR100Train(path, transform=transform_train)
#     cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#     cifar100_training_loader = DataLoader(
#         cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

#     return cifar100_training_loader

# def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
#     """ return training dataloader
#     Args:
#         mean: mean of cifar100 test dataset
#         std: std of cifar100 test dataset
#         path: path to cifar100 test python dataset
#         batch_size: dataloader batchsize
#         num_workers: dataloader num_works
#         shuffle: whether to shuffle
#     Returns: cifar100_test_loader:torch dataloader object
#     """

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#     #cifar100_test = CIFAR100Test(path, transform=transform_test)
#     cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
#     cifar100_test_loader = DataLoader(
#         cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

#     return cifar100_test_loader
