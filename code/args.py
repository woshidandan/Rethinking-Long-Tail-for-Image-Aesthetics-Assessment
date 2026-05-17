import argparse


def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    # path
    parser.add_argument('--csv_path', type=str, default="/root/autodl-tmp/ELTA/AVA/label",
                        help='path to dataset-label csv file')
    parser.add_argument('--dataset_path', type=str, default='/root/autodl-tmp/ELTA/AVA/AVA_dataset/image',
                        help='path to dataset')
    parser.add_argument('--test_dataset_path', type=str, default='/root/autodl-tmp/ELTA/AVA/AVA_dataset/image',
                        help='path to test_dataset')
    parser.add_argument('--generated_dataset_path', type=str,
                        default='/root/autodl-tmp/ELTA/Long-Tail-image-aesthetics-and-quality-assessment-main/ELTA 2.0/code/diffusion/result/AVA/generated',
                        help='path to generated images used by ELTA 2.0 self-training')
    parser.add_argument('--output_dir', type=str, default='runs_ava',
                        help='directory for logs, predictions and checkpoints')

    # params
    parser.add_argument('--loss_type', type=str, choices=['emd', 'mse'], default='emd')
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--start_epoch', default=0, type=int, help='which epoch to start training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_id', type=str, default='1', help='which physical GPU id to expose')
    parser.add_argument('--metric', default='srcc', type=str, choices=['srcc', 'lcc', 'acc'],
                        help='the metric for updating checkpoint')
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help='whether to initialize SwinV2 from timm pretrained weights')
    parser.add_argument('--max_train_batches', type=int, default=None,
                        help='debug option: stop each training epoch after this many batches')
    parser.add_argument('--max_val_batches', type=int, default=None,
                        help='debug option: stop validation/evaluation after this many batches')

    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='the checkpoint to resume training')
    parser.add_argument('--reset_epoch', default=False, action='store_true',
                        help='load checkpoint weights but restart epoch counter and optimizer')
    parser.add_argument('-e', '--evaluate', default=None, dest='evaluate', type=str,
                        help='evaluate and generate pseudo-labels')
    parser.add_argument('--st', default=False, action='store_true',
                        help='whether to enable self-training')
    parser.add_argument('--retrain', default=False, action='store_true',
                        help='whether to use RRT:regressor retraining')
    
    parser.add_argument('--mixup', default=True, action=argparse.BooleanOptionalAction,
                        help='whether to enable TFA feature-level mixup')
    parser.add_argument('--tau_1', type=float, default=0.5)
    parser.add_argument('--tau_2', type=float, default=2)
    parser.add_argument('--simloss_weight', type=float, default=0.0)

    args = parser.parse_args()
    return args
