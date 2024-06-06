import argparse

def args_parse():
    parser = argparse.ArgumentParser(description='KAsyWithFL Implement.')

    # args.attack,dba_class, dataset, trigger, apple

    # 添加命令行参数
    parser.add_argument('-d', '--dataset', type=str,default="FashionMNIST", help='training dataset.(CIFAR100/CIFAR10/FashionMNIST/EMNIST)')
    parser.add_argument('-c', '--clients', type=int,default=100, help='Clients number.')
    parser.add_argument('-k', '--kasync', type=int,default=20, help='kasyncnumber.')
    parser.add_argument('-e', '--epoches', type=int,default=1000, help='Federated epoches time.')
    parser.add_argument('-l', '--lr', type=float, default=0.01, help='Learning rate of each client.')
    parser.add_argument('-b', '--batchsize', type=int, default=64, help='Batch size of client.')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='Parameter of Direchlet distribution.')
    parser.add_argument('-t', '--times', type=int, default=10, help='Local training epoches.')
    parser.add_argument('-n', '--noniid', type=int, default=3, help='Type of NonIID(1-3).')
    parser.add_argument('-o', '--outrate', type=float, default=0.1, help='Out rate of each federated round.')
    parser.add_argument('-he', '--heterogenerity', type=bool, default=True, help='Use model heterogenerity.(True,False)')
    parser.add_argument('-r', '--cliprate', type=float, default=0.7, help='Clip rate if clients.(0.1-1)')
    parser.add_argument('--dev', type=str, default="cuda", help='Use training device.(cuda,cpu)')
    parser.add_argument('--dec', type=str, default="SAFLD", help='Detector(SAFLD,FLD,FLDNorm)')
    parser.add_argument('--logtitle', type=str, default="cmp",help="Experiment log title.")
    ## Attack Part
    parser.add_argument('--malicious_num', type=int, default="1", help='Malicious client num.')
    parser.add_argument('--malicious_type', type=str, default="dba", help='Malicious client type.(badnet,dba)')
    parser.add_argument('--attack', type=str, default="dba", help='Attack type.(dba)')
    parser.add_argument('--dba_class', type=str, default=0.7, help='Distribution backdoor attack class.')
    parser.add_argument('--trigger', type=str, default="square", help='Trigger type(square,pattern,watermark,apple)')
    parser.add_argument('--attack_label', type=int, default=1, help='The task of the backdoor attack is to change Attack_goal to Attack_label.')
    parser.add_argument('--attack_goal', type=int, default=2, help='The task of the backdoor attack is to change Attack_goal to Attack_label(-1 means attack every label).')
    parser.add_argument('--triggerY', type=int, default=1, help='Trigger Y.')
    parser.add_argument('--triggerX', type=int, default=2, help='Trigger X.')
    # 指定水印和苹果图片的地址
    parser.add_argument('--watermark', type=str, default="utils/apple.png", help='Path of watermark image.')
    parser.add_argument('--apple', type=str, default="utils/watermark.png", help='Path of apple image.')
    args = parser.parse_args()
    return args