from utils_general import *
from utils_methods import *
from utils_methods_FedDC import train_FedDC


if __name__ == '__main__':

    # Dataset initialization
    data_path = 'Folder/' # The folder to save Data & Model


    n_client = 100
    # Generate IID or Dirichlet distribution
    # # IID
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
    # unbalanced
    #data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0.3, data_path=data_path)

    # Dirichlet (0.6)
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
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
        

