from utils_general import *
from utils_methods import *
from utils_methods_FedDC import train_FedDC
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='convnet', help='choose from convnet/resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',help='fl algorithms: fedavg/fedprox/scaffold/fednova')
    # parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    # parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    # parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    # parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=10, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=42, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--rootdir', type=str, required=False, default="./result/bench/", help='root log directory path')
    # parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.1, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    # parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    # parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--half_cpu_cores', action='store_true', default=False, help='limit to use 1/2 of all physical cpu cores')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay of the optimizer')
    
    args = parser.parse_args()
    # args.device = 'cuda'
    return args

if __name__ == '__main__':
    args = get_args()

    # Dataset initialization
    args.data_path = 'Folder/' # The folder to save Data & Model



    # Dirichlet (0.3)
    data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)

    model_name = 'cifar10_LeNet' # Model type

    ###
    # Common hyperparameters

    com_amount = 600
    save_period = 200
    weight_decay = 1e-3
    batch_size = 50
    #act_prob = 1
    act_prob = 0.15
    suffix = model_name
    lr_decay_per_round = 0.998

    # Model function
    model_func = lambda : client_model(model_name)
    init_model = model_func()


    # Initalise the model for all methods with a random seed or load it from a saved initial model
    torch.manual_seed(37)
    init_model = model_func()
    if not os.path.exists('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)):
        if not os.path.exists('%sModel/%s/' %(data_path, data_obj.name)):
            print("Create a new directory")
            os.mkdir('%sModel/%s/' %(data_path, data_obj.name))
        torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)))    



    print('FedAvg')

    epoch = 5
    learning_rate = 0.1
    print_per = 5

    [fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_FedAvg(
        data_obj=data_obj, 
        act_prob=act_prob ,
        learning_rate=learning_rate, 
        batch_size=batch_size, 
        epoch=epoch, 
        com_amount=com_amount, 
        print_per=print_per, 
        weight_decay=weight_decay, 
        model_func=model_func, 
        init_model=init_model,
        sch_step=1, 
        sch_gamma=1, 
        save_period=save_period, 
        suffix=suffix, 
        trial=False, 
        data_path=data_path, 
        lr_decay_per_round=lr_decay_per_round
        )
        

