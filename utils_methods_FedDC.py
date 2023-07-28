from utils_libs import *
from utils_dataset import *
from utils_dataset_wy import *
from utils_models import *
from utils_general import *
# from tensorboardX import SummaryWriter
### Methods



def train_FedDC(
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
    n_train_list = [len(client_data_train[i]) for i in range(n_clnt)]
    weight_list = [n/sum(n_train_list) for n in n_train_list]

    fed_model = copy.deepcopy(init_model).to(args.device)
    best_glob_acc = 0
    glob_loss, glob_acc = [], []
    
    selected_clnts = list(range(n_clnt)) # WY to include all clients
    clnt_models = [copy.deepcopy(init_model.to(args.device)) for _ in range(n_clnt)]
    # clnt_models_cache = [copy.deepcopy(init_model.to(args.device)) for _ in range(n_clnt)]

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list  = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par


    n_par = len(get_mdl_params([init_model])[0])
    state_gadient_diffs = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
        
    cur_cld_model = copy.deepcopy(init_model).to(device)
    cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
    
    for i in range(args.comm_round):            
        logger.info(f"Round {i:2d}: selected clients: {selected_clnts}")

        # WY NOTE: loop over clients for local update
        fed_para = fed_model.state_dict()

        delta_g_sum = np.zeros(n_par)
        
        for clnt in selected_clnts:
            logger.info(f'Round {i:2d}: training client {clnt}')

            # client update the running local model by loading the weights of lastly updated global model
            clnt_models[clnt].load_state_dict(copy.deepcopy(fed_para)) 

            # # client also restore the inital model weights of a new round into a cache
            # clnt_models_cache[clnt].load_state_dict(copy.deepcopy(fed_para))
            
            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device) #Theta
            
            local_update_last = state_gadient_diffs[clnt] # delta theta_i
            global_update_last = state_gadient_diffs[-1]/weight_list[clnt] #delta theta
            alpha = alpha_coef / weight_list[clnt] 
            hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device) #h_i
            clnt_models[clnt] = train_model_FedDC_wy(
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
            


            curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
            delta_param_curr = curr_model_par-cld_mdl_param
            parameter_drifts[clnt] += delta_param_curr 
            beta = 1/n_minibatch/args.lr
            
            state_g = local_update_last - global_update_last + beta * (-delta_param_curr) 
            delta_g_cur = (state_g - state_gadient_diffs[clnt])*weight_list[clnt] 
            delta_g_sum += delta_g_cur
            state_gadient_diffs[clnt] = state_g 
            clnt_params_list[clnt] = curr_model_par 
            

        delta_g_cur = 1 / n_clnt * delta_g_sum  
        state_gadient_diffs[-1] += delta_g_cur  
        
        fed_model_new = set_client_from_params(fed_model, np.mean(clnt_params_list, axis = 0))
        fed_model.load_state_dict(fed_model_new)

        cur_cld_model_new = set_client_from_params(cur_cld_model.to(device), cld_mdl_param) 
        
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
        logger.info(f'>> Round {i:2d}: Global model test loss: {loss_tst:.4f}')
        logger.info(f'>> Round {i:2d}: Global model test accuracy: {acc_tst*100:.2f}%, historical best acc: {best_glob_acc*100:.2f}%')
        


        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, cur_cld_model, data_obj.dataset, 0)
        print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
        tst_cur_cld_perf[i] = [loss_tst, acc_tst]
        
        
        writer.add_scalars('Loss/test', 
                {
                    'Sel clients':tst_sel_clt_perf[i][0],
                    'All clients':tst_all_clt_perf[i][0],
                    'Current cloud':tst_cur_cld_perf[i][0]
                }, i
                )
        
        writer.add_scalars('Accuracy/test', 
                {
                    'Sel clients':tst_sel_clt_perf[i][1],
                    'All clients':tst_all_clt_perf[i][1],
                    'Current cloud':tst_cur_cld_perf[i][1]
                }, i
                )     

                    
    return 


