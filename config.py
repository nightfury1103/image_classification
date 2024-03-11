import os

ALPHA = os.getenv('alpha', None)

DATA_DIR = {
    'INTEL':
    {
        'train': 'datasets/INTEL/trained.csv',
        'test': 'datasets/INTEL/seg_test'
    },
    'CIFAR':
    {
        'train': 'datasets/CIFAR/train',
        'test': 'datasets/CIFAR/test'
    }
}