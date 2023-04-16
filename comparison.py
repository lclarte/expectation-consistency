import numpy as np
import sacred
import scipy.optimize as optimize
import torch
import torch.nn.functional as functional

import calibration as uncertaintycalibration
import densenet, resnet
from dataloader import get_dataloader, get_dataloader_noisy_validation_data

exp = sacred.Experiment("exp_calibration")

def load_model(model_path, model_type):
    if "chenyaofo" in model_type:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_path, pretrained=True)

    elif "resnet" in model_path:
        model = {
            'resnet20' : resnet.resnet20,
            'resnet32' : resnet.resnet32,
            'resnet44' : resnet.resnet44,
            'resnet56' : resnet.resnet56,
            'resnet110' : resnet.resnet110,
        }[model_type]()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    elif "densenet" in model_path:
        model = {
            "densenet121" : densenet.densenet121
        }[model_type]()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model

## 

def get_model_ece(logits, labels, num_classes, logits_are_proba = False):
    if not logits_are_proba:
        probas = torch.softmax(logits, dim = 1)
    else:
        probas = logits
    return uncertaintycalibration.get_calibration_error(probas, labels, p = 1, debias = False, mode='top-label')

def get_brier_score(logits, labels):
    target_one_hot = torch.nn.functional.one_hot(labels)
    probs = torch.softmax(logits, dim = 1)
    return torch.mean(torch.sum((probs - target_one_hot)**2, axis=1))

def get_classification_error(logits, labels):
    output = torch.argmax(logits, dim = 1)
    correct = (output == labels).sum().item()
    return 1.0 - correct / list(output.size())[0]

## 

def find_optimal_temperature_temperature_scaling(logits, labels, T_min = 0.01, T_max = 10.0):
    def objective(T):
        probas = logits / T
        return functional.cross_entropy(probas, labels).item()
    
    res = optimize.minimize_scalar(objective, bounds = [T_min, T_max], method = 'bounded')
    return float(res.x)

def find_optimal_temperature_expectation_consistency(logits, labels, T_min = 0.01, T_max = 10.0):
    validation_error = get_classification_error(logits, labels)

    def objective(T):
        probas = torch.max(torch.softmax(logits / T, dim = 1), dim = 1)[0]
        return torch.mean(probas) - (1.0 - validation_error)

    res = optimize.root_scalar(objective, bracket = [T_min, T_max])
    return float(res.root)

## 

def get_logits_and_labels_from_data(model, data_loader):
    model.eval()
    labels_list, logits_list = [], []
    with torch.no_grad():
        for data, labels in data_loader:
            logits_list += model(data).tolist()
            labels_list += labels.tolist()
    return torch.FloatTensor(logits_list), torch.LongTensor(labels_list)
##

@exp.config
def config():
    model_path = "default.bin"
    model_type = "resnet20"
    dataset_str = "cifar10"
    random_seed = 42
    train_ratio = 0.9
    flip_ratio  = 1.0

@exp.automain
def main(model_path, model_type, dataset_str, random_seed, train_ratio, flip_ratio):
    num_classes = {
        "cifar10" : 10,
        "cifar10_noisy" : 10,
        "cifar100": 100,
        "svhn" : 10,
        "svhn_noisy" : 10
    }[dataset_str]

    if 'noisy' in dataset_str:
        _, validation_loader, test_loader = get_dataloader_noisy_validation_data(dataset_str, random_seed, train_ratio, batch_size_train = 128, batch_size_validation = 128, batch_size_test = 128, flip_ratio=flip_ratio)
    else:
        _, validation_loader, test_loader = get_dataloader(dataset_str, random_seed, train_ratio, batch_size_train = 128, batch_size_validation = 128, batch_size_test = 128)
    model = load_model(model_path, model_type)

    validation_logits, validation_labels = get_logits_and_labels_from_data(model, validation_loader)
    test_logits, test_labels             = get_logits_and_labels_from_data(model, test_loader)
    print(f'Classification error : {get_classification_error(test_logits, test_labels)}') 

    temperature_ts = find_optimal_temperature_temperature_scaling(validation_logits, validation_labels)
    temperature_ec = find_optimal_temperature_expectation_consistency(validation_logits, validation_labels)

    test_ece      = get_model_ece(test_logits, test_labels, num_classes)
    test_ece_ts   = get_model_ece(test_logits / temperature_ts, test_labels, num_classes)
    test_ece_ec   = get_model_ece(test_logits / temperature_ec, test_labels, num_classes)
    
    print(f'Temperature of TS and EC : {temperature_ts}, {temperature_ec}')
    print(f'Test EC (Uncal., TS, EC) : {test_ece}, {test_ece_ts}, {test_ece_ec}')
    
    test_brier = get_brier_score(test_logits, test_labels)
    test_brier_ts = get_brier_score(test_logits / temperature_ts, test_labels)
    test_brier_ec = get_brier_score(test_logits / temperature_ec, test_labels)
    print(f'Test BS (Uncal., TS, EC) : {test_brier}, {test_brier_ts}, {test_brier_ec}')

