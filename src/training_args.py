import argparse


def build_training_parser(default_output, iterations=100, train_classes=2,
                          include_set_batches=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-list', required=True, help='training audio/label list')
    parser.add_argument('--valid-list', required=True, help='validation audio/label list')
    parser.add_argument('--ssl-checkpoint', required=True, help='XLS-R 300M checkpoint')
    parser.add_argument('--audio-root', default=None,
                        help='root prepended to relative manifest paths')
    parser.add_argument('-root', '--dataset-root', '--dataset_root', dest='dataset_root',
                        default='../dataset', help='dataset root')
    parser.add_argument('-exp', '--experiment-root', '--experiment_root',
                        dest='experiment_root', default=default_output,
                        help='model and metric output directory')
    parser.add_argument('-nep', '--epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', '--learning_rate',
                        dest='learning_rate', type=float, default=0.001)
    parser.add_argument('-lrS', '--lr-scheduler-step', '--lr_scheduler_step',
                        dest='lr_scheduler_step', type=int, default=20)
    parser.add_argument('-lrG', '--lr-scheduler-gamma', '--lr_scheduler_gamma',
                        dest='lr_scheduler_gamma', type=float, default=0.5)
    parser.add_argument('-its', '--iterations', type=int, default=iterations)
    parser.add_argument('-cTr', '--classes-per-it-tr', '--classes_per_it_tr',
                        dest='classes_per_it_tr', type=int, default=train_classes)
    parser.add_argument('-nsTr', '--num-support-tr', '--num_support_tr',
                        dest='num_support_tr', type=int, default=5)
    parser.add_argument('-nqTr', '--num-query-tr', '--num_query_tr',
                        dest='num_query_tr', type=int, default=15)
    parser.add_argument('-cVa', '--classes-per-it-val', '--classes_per_it_val',
                        dest='classes_per_it_val', type=int, default=2)
    parser.add_argument('-nsVa', '--num-support-val', '--num_support_val',
                        dest='num_support_val', type=int, default=5)
    parser.add_argument('-nqVa', '--num-query-val', '--num_query_val',
                        dest='num_query_val', type=int, default=15)
    parser.add_argument('-seed', '--manual-seed', '--manual_seed',
                        dest='manual_seed', type=int, default=7)
    parser.add_argument('--cuda', action='store_true')

    if include_set_batches:
        parser.add_argument('-set_trn', '--num-sets-per-class-tr',
                            '--num_sets_per_class_tr', dest='num_sets_per_class_tr',
                            type=int, default=4)
        parser.add_argument('-set_val', '--num-sets-per-class-val',
                            '--num_sets_per_class_val', dest='num_sets_per_class_val',
                            type=int, default=4)
    return parser


def add_rawboost_arguments(parser):
    parser.add_argument('--algo', type=int, default=0)
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)
    return parser
