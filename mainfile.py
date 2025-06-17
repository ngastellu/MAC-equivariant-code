from utils import *
import torch
from torchsummary import summary
import gc
from time import perf_counter


def main(configs):
    rundir = configs.experiment_name
    if not os.path.isdir(rundir):
        os.makedirs(rundir)

    converged = 0
    epoch = 1
    tr_err_hist = []
    te_err_hist = []
# experiment = get_comet_experiment(configs)
    print(f'initialize_training...', end=' ', flush=True)
    start = perf_counter()
    model, optimizer, dataDims = initialize_training(configs)
    end = perf_counter()
    print(f'Done! [{end-start} seconds]', flush=True)

    if configs.start_epoch != 0:
        model, optimizer, epoch = load_checkpoint_training(configs, rundir, model, optimizer)
    #model.cpu()


   # gc.collect()
    torch.cuda.empty_cache()

    #   log_input_stats(configs, experiment, input_analysis)

    print('Imported and Analyzed Training Dataset {}'.format(configs.training_dataset), flush=True)

    # if configs.CUDA:
    #     backends.cudnn.benchmark = True  # auto-optimizes certain backend processes
    #     model = nn.DistributedDataParallel(model)  # go to multi-GPU training
    #     print("Using", torch.cuda.device_count(), "GPUs")
    #     model.to(torch.device("cuda:0"))
    #     #print(summary(model, [(dataDims['channels'], dataDims['input y dim'], dataDims['input x dim'])]))

    ## BEGIN TRAINING/GENERATION
    if configs.max_epochs == 0:  # no training, just samples
        sample, time_ge = generation(configs, dataDims, model)
       # log_generation_stats(experiment, sample, agreements, output_analysis)

    else:  # train it AND make samples!

        #if configs.auto_training_batch:
        ##    configs.training_batch_size, changed = get_training_batch_size(configs, model)  # confirm we can keep on at this batch size
        #else:
        #    changed = 0
        #if changed == 1:  # if the training batch is different, we have to adjust our batch sizes and dataloaders
        #    tr, te, _ = get_dataloaders(configs)
        #    print('Training batch set to {}'.format(configs.training_batch_size))
        #else:

        print(f'get_dataloaders...', end=' ', flush=True)
        start = perf_counter()
        tr, te, _ = get_dataloaders(configs)
        end = perf_counter()
        print(f'Done! [{end-start} seconds]', flush=True)

       # print(['tr and te',len(tr),len(te)])
    #    print([configs.training_batch_size]) 
        if epoch  == 1:
            logfile = os.path.join(rundir, configs.experiment_name + '.log')
        else:
            logfile = os.path.join(rundir, configs.experiment_name + f'_restart_epoch-{epoch}' + '.log') 
        f = open(logfile, 'w')

        # ------- BEGIN TRAINING LOOP -------
        print('******* Entering training loop *******', flush=True)
        while (epoch <= (configs.max_epochs + 1)) & (converged == 0):  # over a certain number of epochs or until converged

            print(f'\n[epoch #{epoch}] model_epoch (train)...', end=' ', flush=True)
            start=perf_counter()
            err_tr, time_tr = model_epoch(configs, dataDims=dataDims, trainData=tr, model=model, optimizer=optimizer, update_gradients=True)  # train & compute loss
            end = perf_counter()
            print(f'Done! [{end-start} seconds]', flush=True)

            print(f'[epoch #{epoch}] model_epoch (test)...', end=' ', flush=True)
            start=perf_counter()
            err_te, time_te = model_epoch(configs, dataDims=dataDims, trainData=te, model=model, update_gradients=False)  # compute loss on test set
            end = perf_counter()
            print(f'Done! [{end-start} seconds]', flush=True)

            tr_err_hist.append(torch.mean(torch.stack(err_tr)))
            te_err_hist.append(torch.mean(torch.stack(err_te)))
            print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_tr, time_te),flush=True)
            converged = auto_convergence(configs, epoch, tr_err_hist, te_err_hist)
            if int(epoch % 2 == 0):
                f.write(str(epoch) + " " + str(torch.mean(torch.stack(err_tr))) + " " + str(time_tr) + " " + str(torch.mean(torch.stack(err_te)))+'\n')

            if epoch % configs.generation_period == 0:
                sample, time_ge= generation(configs, dataDims, model,epoch)
               # log_generation_stats(configs, epoch, experiment, sample, agreements, output_analysis)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},os.path.join(rundir, f'model-amorphous-gen-{epoch}.pt'))
                sample, time_ge = generation(configs, dataDims, model,epoch)



            if epoch%100==0:
                print('Saving model...', end=' ', flush=True)
                start = perf_counter()
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},os.path.join(rundir, f'model-epoch_{epoch}.pt'))
                end = perf_counter()
                print(f'Done! [{end-start} seconds]', flush=True)



            epoch += 1

        # generate samples
        f.close()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},os.path.join(rundir, f'model-amorphous-{configs.experiment_name}.pt'))
    #    sample, time_ge = generation(configs, dataDims, model,epoch)
       # log_generation_stats(configs, epoch, experiment, sample, agreements, output_analysis)
    print('finished!')
