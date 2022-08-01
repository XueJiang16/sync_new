from utils import log
import resnetv2
import torch
import torchvision as tv
import time

import numpy as np
import random
from utils.test_utils import arg_parser, get_measures
import os
import glob

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score
import resnet

import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import mmcls.models

def model_forward(model, data):
    if isinstance(model, mmcls.models.classifiers.image.ImageClassifier):
        return model(return_loss=False, img=data, img_metas=None, softmax=False, post_process=False)
    return model(data)

def iterate_data_msp(data_loader, model):
    confs = []
    cls = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model_forward(model, x)
            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data)
            cls.extend(y)
        if b % 100 == 0:
            print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_msp_custom(data_loader, model, targets):
    confs = []
    cls = []
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model_forward(model, x)
            # sim = torch.abs(m(logits) - targets)
            softmax_output = m(logits)

            sim1 = softmax_output - targets
            sim2 = -softmax_output * targets
            sim2 = sim2.sum(1) /(torch.norm(softmax_output, dim=1) * torch.norm(targets, dim=1))
            sim2 = sim2.unsqueeze(1)
            # sim = sim2*softmax_output
            sim = sim1
            conf, _ = torch.max(sim, dim=-1)
            # conf = sim
            confs.extend(conf.data)
            cls.extend(y)
        if b % 100 == 0:
            print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()



def iterate_data_odin(data_loader, model, epsilon, temper):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    cls = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs, _ = model_forward(model, x)
        cls.extend(y)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs, _ = model_forward(model, Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_odin_custom(data_loader, model, epsilon, temper, targets, mode='linear'):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    cls = []
    targets = np.expand_dims(targets, axis=0)
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs, _ = model_forward(model, x)
        softmax_output = m(outputs)
        softmax_output = softmax_output.data.cpu()
        softmax_output = softmax_output.numpy()
        cls.extend(y)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs, _ = model_forward(model, Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        sim = -softmax_output * targets
        sim = sim.sum(axis=1) / (np.linalg.norm(nnOutputs, axis=-1) * np.linalg.norm(targets, axis=-1))
        sim = np.expand_dims(sim, axis=1)
        nnOutputs = sim * nnOutputs
        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_energy(data_loader, model, temper):
    confs = []
    cls = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model_forward(model, x)
            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data)
            cls.extend(y)
            if b % 100 == 0:
                print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_energy_custom(data_loader, model, temper, targets, mode='linear'):
    confs = []
    cls = []
    # targets = torch.ones(1000)
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model_forward(model, x)
            conf = temper * torch.logsumexp(logits / temper, dim=1) #(batch)

            softmax_output = m(logits)
            sim = -softmax_output * targets
            sim = sim.sum(1) / (torch.norm(softmax_output, dim=1) * torch.norm(targets, dim=1))
            # targets_en = temper * torch.logsumexp(targets / temper, dim=-1)
            conf = conf * sim
            # conf = torch.abs(conf - targets)
            confs.extend(conf.data)
            cls.extend(y)
            if b % 100 == 0:
                print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor):
    confs = []
    cls = []
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        # if b == 10:
        #     break
        x = x.cuda()
        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
        cls.extend(y)
        # assert len(confs) == len(cls), print(b)
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_gradnorm(data_loader, model, temperature, num_classes):
    confs = []
    cls = []
    with torch.no_grad():
        for b, (x, y) in enumerate(data_loader):
            if b % 10 == 0:
                print('{} batches processed'.format(b))
            inputs = Variable(x.cuda(), requires_grad=False)
            outputs, features = model_forward(model, inputs)
            U = torch.norm(features, p=1, dim=1)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            V = torch.norm((1 - num_classes * out_softmax), p=1, dim=1)
            S = U * V / 2048 / num_classes
            confs.extend(S)
            cls.extend(y)
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()
# def iterate_data_gradnorm(data_loader, model, temperature, num_classes):
#     import pickle
#     confs = []
#     cls = []
#     filename2score = dict()
#     with torch.no_grad():
#         for b, (x, y, name) in enumerate(data_loader):
#             if b % 10 == 0:
#                 print('{} batches processed'.format(b))
#             inputs = Variable(x.cuda(), requires_grad=False)
#             outputs, features = model_forward(model, inputs)
#             U = torch.norm(features, p=1, dim=1)
#             out_softmax = torch.nn.functional.softmax(outputs, dim=1)
#             V = torch.norm((1 - num_classes * out_softmax), p=1, dim=1)
#             S = U * V / 2048 / num_classes
#             confs.extend(S)
#             cls.extend(y)
#             filename2score[name[0]] = float(outputs.cpu().mean().data)
#             if b == 1000:
#                 break
#     with open("dump_output.pkl", "wb") as f:
#         pickle.dump(filename2score, f)
#     return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()



def iterate_data_newplus2(data_loader, model, temperature, num_classes, targets):
    #reweight
    confs = []
    cls = []
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    with torch.no_grad():
        for b, (x, y) in enumerate(data_loader):
            if b % 10 == 0:
                print('{} batches processed'.format(b))
            inputs = Variable(x.cuda(), requires_grad=False)
            outputs, features = model_forward(model, inputs)
            U = torch.norm(features, p=1, dim=1)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            V = torch.norm((targets - out_softmax), p=1, dim=1)
            S = U * V / 2048 / num_classes
            sim_ = -out_softmax * targets
            sim = sim_.sum(1) / (torch.norm(out_softmax, dim=1) * torch.norm(targets, dim=1))
            S = S*sim
            confs.extend(S)
            cls.extend(y)
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_newplus(data_loader, model, temperature, num_classes, targets):
    confs = []
    cls = []
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    with torch.no_grad():
        for b, (x, y) in enumerate(data_loader):
            if b % 10 == 0:
                print('{} batches processed'.format(b))
            inputs = Variable(x.cuda(), requires_grad=False)
            outputs, features = model_forward(model, inputs)
            U = torch.norm(features, p=1, dim=1)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            V = torch.norm((targets - out_softmax), p=1, dim=1)
            S = U * V / 2048
            confs.extend(S)
            cls.extend(y)
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_gradnorm_o(data_loader, model, temperature, num_classes):
    confs = []
    labels = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)
        model.zero_grad()
        outputs, _ = model_forward(model, inputs)
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature

        loss = torch.sum(torch.mean(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()

        # layer_grad = model.head.conv.weight.grad.data
        if isinstance(model, mmcls.models.classifiers.image.ImageClassifier):
            layer_grad = model.head.layers[1].fc.weight.grad.data
        else:
            layer_grad = model.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad))
        confs.append(layer_grad_norm)
        label = y.clone().detach()
        labels.append(label)
        # print('original:',layer_grad_norm)
    return torch.tensor(confs).cuda(), torch.tensor(labels).cuda()

def iterate_data_new(data_loader, model, temperature, num_classes,target):
    confs = []
    labels = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()
        outputs, _ = model_forward(model, inputs)
        target = torch.tensor(target)
        targets = target.unsqueeze(dim=0).cuda()
        outputs = outputs / temperature
        loss = torch.sum(-targets * logsoftmax(outputs), dim=-1)
        loss.backward()
        if isinstance(model, mmcls.models.classifiers.image.ImageClassifier):
            layer_grad = model.head.layers[1].fc.weight.grad.data
        else:
            layer_grad = model.fc.weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad)) / 2048 #fc dimension
        confs.append(layer_grad_norm)
        label = y.clone().detach()
        labels.append(label)
    # exit()
    return torch.tensor(confs).cuda(), torch.tensor(labels).cuda()

def iterate_data_cosine(data_loader, model, targets):
    confs = []
    cls = []
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    m = torch.nn.Softmax(dim=-1).cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model_forward(model, x)
            # softmax_output = m(logits)
            # sim = -softmax_output * targets
            # sim = sim.sum(1) /(torch.norm(softmax_output, dim=1) * torch.norm(targets, dim=1))
            # sim = sim.unsqueeze(1)
            outputs = logits
            sim = torch.sum(-targets * logsoftmax(outputs), dim=-1)
            conf = sim
            confs.extend(conf.data)
            cls.extend(y)
        if b % 100 == 0:
            print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_cosnorm(data_loader, model, targets):
    confs = []
    labels = []
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()
        logits, _ = model_forward(model, inputs)
        softmax_output = m(logits)
        sim = -softmax_output * targets
        sim = sim.sum(1) / (torch.norm(softmax_output, dim=1) * torch.norm(targets, dim=1))
        loss = sim.unsqueeze(1)
        loss.backward()

        # layer_grad = model.head.conv.weight.grad.data
        if isinstance(model, mmcls.models.classifiers.image.ImageClassifier):
            layer_grad = model.head.layers[1].fc.weight.grad.data
        else:
            layer_grad = model.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad))
        confs.append(layer_grad_norm)
        label = y.clone().detach()
        labels.append(label)
        # print('original:',layer_grad_norm)
    return torch.tensor(confs).cuda(), torch.tensor(labels).cuda()

def iterate_data_confidence(data_loader, model, isID=True):
    confs = []
    labels = []
    softmax = torch.nn.Softmax(dim=-1).cuda()
    with torch.no_grad():
        if isID:
            for b, (x, y) in enumerate(data_loader):
                if b % 100 == 0:
                    print('{} batches processed'.format(b))
                inputs = x.cuda()
                # model.zero_grad()
                outputs, _ = model(inputs)
                prob = softmax(outputs).cpu().numpy()
                confs.append(prob)
                labels.append(y)
            return np.concatenate(confs, axis=0), np.array(labels)
        else:
            for b, (x, y) in enumerate(data_loader):
                if b % 100 == 0:
                    print('{} batches processed'.format(b))
                inputs =x.cuda()
                outputs, _ = model(inputs)
                prob = softmax(outputs).cpu().numpy()
                confs.append(prob)
        return np.concatenate(confs, axis=0)


def sample_function(a, num_img=128000, m=1):
    f = lambda x: a*(m**a) / x**(a+1)
    upper = 2.2
    lower = 1
    x = np.linspace(lower, upper, num=1000)
    y = f(x)
    scale_factor = num_img/sum(y)
    y = np.round(y * scale_factor)
    return y

def generate_dis_func(res):
    info = []
    with open('./meta/train_labeled.txt', 'r') as f:
        ls = f.readlines()
        for l in ls:
            name, label = l.strip().split(' ')
            info.append(int(label))
    label_stat = Counter(info)
    order = sorted(label_stat.items(), key=lambda s: (-s[1]))
    orders = []
    for item in order:
        orders.append(item[0])
    cls_num = [-1 for _ in range(1000)]
    for i in range(1000):
        cat_idx = orders[i]
        cat_num = int(res[i])
        cls_num[cat_idx] = cat_num
        assert cat_num <= label_stat[cat_idx]
    return cls_num
