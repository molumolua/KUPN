import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="last-fm", help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--batch_size_kg', type=int, default=1024, help='batch size for kg')
    parser.add_argument('--batch_size_cl', type=int, default=1024, help='batch size for cl')
    parser.add_argument('--test_batch_size', type=int, default=128, help='batch size for test')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20,10]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")

    parser.add_argument("--keep_rate",type=float,default=0.1,help="keep rate for extra edges")
    parser.add_argument("--method",type=str,default="add",help="convolution method for user, add or stack")
    parser.add_argument("--drop_learn",type=bool,default=False,help="use drop learn or not")
    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    # ===== contrast and drop learner===== #
    # parser.add_argument("--tau_prefer",type=float,default=0.7)
    # parser.add_argument("--tau_kg",type=float,default=1.5)
    parser.add_argument("--tau_cl",type=float,default=0.7)
    parser.add_argument("--cl_alpha",type=float,default=0.1)
    # parser.add_argument("--K2",type=int,default=2)
    # parser.add_argument("--K3",type=int,default=1)
    parser.add_argument("--neighs", default='[2,1]')
    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()
