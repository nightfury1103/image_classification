import os

ALPHA = os.getenv('alpha', None)

DATA_DIR = {
    'INTEL':
    {
        'train': 'datasets/INTEL/seg_train/seg_train/trained.csv',
        'test': 'datasets/INTEL/seg_test'
    },
    'CIFAR':
    {
        'train': 'datasets/CIFAR/train',
        'test': 'datasets/CIFAR/test'
    }
}