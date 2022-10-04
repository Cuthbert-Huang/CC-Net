from networks.VNet import VNet, CCNet3d_V1

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "ccnet3d_v1" and mode == "train":
        net = CCNet3d_V1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "ccnet3d_v1" and mode == "test":
        net = CCNet3d_V1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net
