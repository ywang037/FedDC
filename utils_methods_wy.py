from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *
# from tensorboardX import SummaryWriter

# helper functions
import time
from utils_libs_wy import *
from utils_dataset_wy import *

def mdl_agg(clnt_models, fed_para, weight_list):
    for clnt in range(len(clnt_models)):
        net_para = clnt_models[clnt].cpu().state_dict()
        if clnt == 0:
            for key in net_para:
                fed_para[key] = net_para[key] * weight_list[clnt]
        else:
            for key in net_para:
                fed_para[key] += net_para[key] * weight_list[clnt]
    return fed_para

### Methods
def train_FedAvg(
        args,
        logger,
        init_model,
        client_data_train,
        test_data,
    ):
    time_start = time.time()
    n_clnt = args.n_parties
    local_trainloaders = [DataLoader(client_data_train[i], batch_size=args.batch_size, shuffle=True) for i in range(args.n_parties)]
    central_testloader = DataLoader(test_data, batch_size=128, shuffle=False)

    # weights for aggregation
    n_train_list = [len(client_data_train[i]) for i in range(n_clnt)]
    weight_list = [n/sum(n_train_list) for n in n_train_list]

    fed_model = copy.deepcopy(init_model).to(args.device)
    best_glob_acc = 0
    glob_loss, glob_acc = [], []
    
    selected_clnts = list(range(n_clnt)) # WY to include all clients
    clnt_models = [copy.deepcopy(init_model.to(args.device)) for _ in range(n_clnt)]

    for i in range(args.comm_round):            
        logger.info(f"Round {i:2d}: selected clients: {selected_clnts}")
        
        # WY NOTE: loop over clients for local update
        fed_para = fed_model.state_dict()
        for clnt in selected_clnts:
            logger.info(f'Round {i:2d}: training client {clnt}')
        
            
            clnt_models[clnt].load_state_dict(copy.deepcopy(fed_para))
            updated_weights = train_model_wy(
                args,
                model=clnt_models[clnt], 
                local_trainloader=local_trainloaders[clnt], 
                testloader=central_testloader
            )
            clnt_models[clnt].load_state_dict(updated_weights)

        fed_para_new = mdl_agg(clnt_models, fed_para, weight_list)
        fed_model.load_state_dict(fed_para_new)
        
        
        # test performance of all clients over centralized test dataset
        # WY NOTE: this is the only accuracy you need to keep and monitor
        acc_tst, loss_tst = evaluation(fed_model, central_testloader, args.device)
        glob_loss.append(loss_tst)
        glob_acc.append(acc_tst)
        
        if acc_tst>best_glob_acc:
            best_glob_acc = acc_tst
            torch.save(
                {
                    'model_checkpoint': fed_model.state_dict(), 
                },
                os.path.join(args.logdir, 'checkpoint_fed_mdl.pt')
            )
            logger.info(f'>> Round {i:2d}: global model checkpoint saved')
                
        # tst_perf_all[i] = [loss_tst, acc_tst]
        logger.info(f'>> Round {i:2d}: Global model test loss: {loss_tst:.4f}')
        logger.info(f'>> Round {i:2d}: Global model test accuracy: {acc_tst*100:.2f}%, historical best acc: {best_glob_acc*100:.2f}%')
        # print(f"**** Communication round {i:2d}, Test Accuracy: {acc_tst:.4f}, Loss: {loss_tst:.4f}")
        
    record_accuracy(args.exp_dir, args.seed, best_glob_acc)
    torch.save(
        {
            'global_model_test_loss': np.array(glob_loss),
            'global_model_test_acc': np.array(glob_acc),
        },
        os.path.join(args.logdir, f'{args.dataset}-{args.model}-learning_curve-seed{args.seed}.pt')
    )
    
    time_end = time.time()
    time_end_stamp = time.strftime('%Y-%m-%d %H:%M:%S') # time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
    sesseion_time = int((time_end-time_start)/60)   
    logger.info('\nSession completed at {}, time elapsed: {} mins. That\'s all folks.'.format(time_end_stamp, sesseion_time))

    return



def train_FedDC_wy(
        args,
        logger,
        init_model,
        client_data_train,
        test_data,
        n_minibatch,
        alpha_coef
    ):

    time_start = time.time()
    n_clnt = args.n_parties
    local_trainloaders = [DataLoader(client_data_train[i], batch_size=args.batch_size, shuffle=True) for i in range(args.n_parties)]
    central_testloader = DataLoader(test_data, batch_size=128, shuffle=False)

    # weights for aggregation
    weight_list = np.asarray([len(client_data_train[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    # n_train_list = [len(client_data_train[i]) for i in range(n_clnt)]
    # weight_list = [n/sum(n_train_list) for n in n_train_list]

    fed_model = copy.deepcopy(init_model).to(args.device)
    best_glob_acc = 0
    glob_loss, glob_acc = [], []
    
    selected_clnts = list(range(n_clnt)) # WY to include all clients
    clnt_models = [copy.deepcopy(init_model.to(args.device)) for _ in range(n_clnt)]
    # clnt_models_cache = [copy.deepcopy(init_model.to(args.device)) for _ in range(n_clnt)]

    n_par = len(get_mdl_params([init_model])[0])
    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list  = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par    
    state_gadient_diffs = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
        
    cur_cld_model = copy.deepcopy(init_model).to(device)
    cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
    
    for i in range(args.comm_round):            
        logger.info(f"Round {i:2d}: selected clients: {selected_clnts}")

        # WY NOTE: loop over clients for local update
        # fed_para = fed_model.state_dict()
        global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device) #Theta
        delta_g_sum = np.zeros(n_par)
        
        for clnt in selected_clnts:
            logger.info(f'Round {i:2d}: training client {clnt}')

            # client update the running local model by loading the weights of lastly updated global model with local drift correction
            clnt_models[clnt].load_state_dict(copy.deepcopy(cur_cld_model.state_dict()))         
            local_update_last = state_gadient_diffs[clnt] # delta theta_i
            global_update_last = state_gadient_diffs[-1]/weight_list[clnt] #delta theta
            alpha = alpha_coef / weight_list[clnt] 
            hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device) #h_i
            updated_weights = train_model_FedDC_wy(
                args,
                clnt_models[clnt], 
                local_trainloaders[clnt], 
                central_testloader,
                alpha,
                local_update_last, 
                global_update_last,
                global_mdl, 
                hist_i
            )
            clnt_models[clnt].load_state_dict(updated_weights)
            curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0] # get the parameter of updated local model
            delta_param_curr = curr_model_par-cld_mdl_param
            parameter_drifts[clnt] += delta_param_curr 
            beta = 1/n_minibatch/args.lr
            
            state_g = local_update_last - global_update_last + beta * (-delta_param_curr) 
            delta_g_cur = (state_g - state_gadient_diffs[clnt])*weight_list[clnt] 
            delta_g_sum += delta_g_cur
            state_gadient_diffs[clnt] = state_g 
            clnt_params_list[clnt] = curr_model_par # list of parameters of updated local model
            
        delta_g_cur = 1 / n_clnt * delta_g_sum  
        state_gadient_diffs[-1] += delta_g_cur 
         
        fed_model_param = np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0) # do model parameter weighted average: model aggregation
        fed_model = set_client_from_params(fed_model, fed_model_param) # normal fedavg global model without local drift correction
        
        loc_param_drift_correction = np.sum(parameter_drifts*weight_list/np.sum(weight_list), axis = 0)
        cld_mdl_param = fed_model_param + loc_param_drift_correction
        cur_cld_model = set_client_from_params(cur_cld_model.to(device), cld_mdl_param) # update the parameters of current cloud model

        acc_tst, loss_tst = evaluation(cur_cld_model, central_testloader, args.device)
        glob_loss.append(loss_tst)
        glob_acc.append(acc_tst)

        if acc_tst>best_glob_acc:
            best_glob_acc = acc_tst
            torch.save(
                {
                    'model_checkpoint': fed_model.state_dict(), 
                },
                os.path.join(args.logdir, 'checkpoint_fed_mdl.pt')
            )
            logger.info(f'>> Round {i:2d}: global model checkpoint saved')
        logger.info(f'>> Round {i:2d}: Global model test loss: {loss_tst:.4f}')
        logger.info(f'>> Round {i:2d}: Global model test accuracy: {acc_tst*100:.2f}%, historical best acc: {best_glob_acc*100:.2f}%')

    record_accuracy(args.exp_dir, args.seed, best_glob_acc)
    torch.save(
        {
            'global_model_test_loss': np.array(glob_loss),
            'global_model_test_acc': np.array(glob_acc),
        },
        os.path.join(args.logdir, f'{args.dataset}-{args.model}-learning_curve-seed{args.seed}.pt')
    )
    
    
    time_end = time.time()
    time_end_stamp = time.strftime('%Y-%m-%d %H:%M:%S') # time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
    sesseion_time = int((time_end-time_start)/60)   
    logger.info('\nSession completed at {}, time elapsed: {} mins. That\'s all folks.'.format(time_end_stamp, sesseion_time))

                    
    return 


