import os
import torch
import torchvision.models as torchvision_models
from checkpoint.bs_models import vit, resnet18
from checkpoint.model import ARTransformer
from checkpoint.fsra import fsra_three_view_net
from checkpoint.rknet import two_view_net, rknet_three_view_net


def load_resnet(args):
    """
    load resnet18.
    :param args:
    :return:
    """
    model = resnet18(num_classes1=args.num_classes1, num_classes2=args.num_classes2)

    # load from resume, start training from a certain epoch
    if args.model1_resume:
        if os.path.isfile(args.model1_resume):
            print("=> loading checkpoint '{}'".format(args.model1_resume))
            checkpoint = torch.load(args.model1_resume, map_location='cpu')

            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict
            for key in list(state_dict.keys()):
                new_state_dict[key[len("module."):]] = state_dict[key]
                del new_state_dict[key]
            model.load_state_dict(new_state_dict, strict=False)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            best_acc3 = checkpoint['best_acc3']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            print("best_acc3: " + str(best_acc3))

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model1_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model1_resume))

    return model


def load_vit(args):
    """
    load vit.
    :param args:
    :return:
    """
    model = vit(num_classes1=args.num_classes1, num_classes2=args.num_classes2)

    # load from resume, start training from a certain epoch
    if args.model2_resume:
        if os.path.isfile(args.model2_resume):
            print("=> loading checkpoint '{}'".format(args.model2_resume))
            checkpoint = torch.load(args.model2_resume, map_location='cpu')

            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict
            for key in list(state_dict.keys()):
                new_state_dict[key[len("module."):]] = state_dict[key]
                del new_state_dict[key]
            model.load_state_dict(new_state_dict, strict=False)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            best_acc3 = checkpoint['best_acc3']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            print("best_acc3: " + str(best_acc3))

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model2_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model2_resume))

    return model


# def load_ibl():
#     model = torch.hub.load('../.cache/torch/hub/yxgeee_OpenIBL_master', model='vgg16_netvlad',
#                            pretrained=True, source='local').eval()
#     return model


def load_rknet():
    """
    load rknet.
    :return:
    """
    model = rknet_three_view_net(701, droprate=0.5, VGG16=False)
    model.load_state_dict(torch.load("checkpoint/rknet.pth"), strict=False)
    return model


def load_fsra():
    """
    load fsra.
    :return:
    """
    model = fsra_three_view_net(701, block=3)
    model.load_state_dict(torch.load("checkpoint/fsra.pth"), strict=False)
    return model


def load_our_model(args):
    """
    load our angle robustness model.
    :param args:
    :return:
    """
    backbone = torchvision_models.mobilenet_v3_small(pretrained=True)

    model = ARTransformer(
        backbone=backbone,
        extractor_dim=576,
        num_classes1=args.num_classes1,
        num_classes2=args.num_classes2,
        len=args.len,
        dim=512,
        depth=6,
        heads=8,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    )

    # load from resume, start training from a certain epoch
    if args.our_model_resume:
        if os.path.isfile(args.our_model_resume):
            print("=> loading checkpoint '{}'".format(args.our_model_resume))
            checkpoint = torch.load(args.our_model_resume, map_location='cpu')

            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict
            for key in list(state_dict.keys()):
                new_state_dict[key[len("module."):]] = state_dict[key]
                del new_state_dict[key]
            model.load_state_dict(new_state_dict, strict=False)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            best_acc3 = checkpoint['best_acc3']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            print("best_acc3: " + str(best_acc3))

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.our_model_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.our_model_resume))

    return model
