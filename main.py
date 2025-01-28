"""
    Train networks on 1d hierarchical models of data.
"""

import os
import argparse
import time
import math
import pickle
import sklearn
import numpy
from models import *
import copy
from functools import partial
import matplotlib.pyplot as plt
from init import init_fun
from optim_loss import loss_func, regularize, opt_algo, measure_accuracy
from utils import cpu_state_dict, args2train_test_sizes
from observables import locality_measure, state2permutation_stability, state2clustering_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

def plot(time, mse_values1, mse_values2, mse_values3, mse_values4, mse_values5, mse_values6, mse_values7, title):

    if len(time) != len(mse_values1):
        raise ValueError("The length of time and mse_values must be the same.")

    fig, ax = plt.subplots()

    ax.plot(time, mse_values1, marker='o', linestyle='-', color='b', label="layer 1 features")
    ax.plot(time, mse_values2, marker='o', linestyle='-', color='r', label="layer 2 features")
    ax.plot(time, mse_values3, marker='o', linestyle='-', color='g', label="layer 3 features")
    ax.plot(time, mse_values4, marker='o', linestyle='-', color='c', label="layer 4 features")
    ax.plot(time, mse_values5, marker='o', linestyle='-', color='k', label="Class label of RHM")
    ax.plot(time, mse_values6, marker='o', linestyle='-', color='m', label="Hidden feature of RHM(shallow)")
    ax.plot(time, mse_values7, marker='o', linestyle='-', color='orange', label="Hidden feature of RHM(deep)")

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')

    ax.grid(True)

    plt.legend()

    plt.show()

def run(args):

    # if args.dtype == 'float64':
    #     torch.set_default_dtype(torch.float64)
    # if args.dtype == 'float32':
    #     torch.set_default_dtype(torch.float32)
    # if args.dtype == 'float16':
    #     torch.set_default_dtype(torch.float16)

    best_acc = 0  # best test accuracy
    criterion = partial(loss_func, args)

    trainloader, testloader, net0 = init_fun(args)

    # scale batch size when larger than train-set size
    if (args.batch_size >= args.ptr) and args.scale_batch_size:
        args.batch_size = args.ptr // 2

    if args.save_dynamics:
        dynamics = [{"acc": 0.0, "epoch": 0., "net": cpu_state_dict(net0)}]
    else:
        dynamics = None

    loss = []
    terr = []
    locality = []
    stability = []
    clustering_error = []
    epochs_list = []

    best = dict()
    trloss_flag = 0

    for net, epoch, losstr, avg_epoch_time in train(args, trainloader, net0, criterion):

        assert str(losstr) != "nan", "Loss is nan value!!"
        loss.append(losstr)
        epochs_list.append(epoch)

        # measuring locality for fcn nets
        if args.locality == 1:
            assert args.net == 'fcn', "Locality can only be computed for fcns !!"
            state = net.state_dict()
            hidden_layers = [state[k] for k in state if 'w' in k][:-2]
            with torch.no_grad():
                locality.append(locality_measure(hidden_layers, args)[0])

        # measure stability to semantically equivalent data realizations
        if args.stability == 1:
            state = net.state_dict()
            stability.append(state2permutation_stability(state, args))
        if args.clustering_error == 1:
            state = net.state_dict()
            clustering_error.append(state2clustering_error(state, args))

        if epoch % 10 != 0 and not args.save_dynamics: continue

        if testloader:
            acc = test(args, testloader, net, criterion, print_flag=epoch % 5 == 0)
        else:
            acc = torch.nan
        terr.append(100 - acc)

        if args.save_dynamics:
        #     and (
        #     epoch
        #     in (10 ** torch.linspace(-1, math.log10(args.epochs), 30)).int().unique()
        # ):
            # save dynamics at 30 log-spaced points in time
            dynamics.append(
                {"acc": acc, "epoch": epoch, "net": cpu_state_dict(net)}
            )
        if acc > best_acc:
            best["acc"] = acc
            best["epoch"] = epoch
            if args.save_best_net:
                best["net"] = cpu_state_dict(net)
            # if args.save_dynamics:
            #     dynamics.append(best)
            best_acc = acc
            print(f"BEST ACCURACY ({acc:.02f}) at epoch {epoch:.02f} !!", flush=True)

        out = {
            "args": args,
            "epoch": epochs_list,
            "train loss": loss,
            "terr": terr,
            "locality": locality,
            "stability": stability,
            "clustering_error": clustering_error,
            "dynamics": dynamics,
            "best": best,
        }

        yield out

        if (losstr == 0 and args.loss == 'hinge') or (losstr < args.zero_loss_threshold and args.loss == 'cross_entropy'):
            trloss_flag += 1
            if trloss_flag >= args.zero_loss_epochs:
                pass
                #break

    try:
        wo = weights_evolution(net0, net)
    except:
        print("Weights evolution failed!")
        wo = None

    if args.locality == 2:
        assert args.net == 'fcn', "Locality can only be computed for fcns !!"
        state = net.state_dict()
        hidden_layers = [state[k] for k in state if 'w' in k][:-2]
        with torch.no_grad():
            locality.append(locality_measure(hidden_layers, args)[0])

    if args.stability == 2:
        state = net.state_dict()
        stability.append(state2permutation_stability(state, args))

    if args.clustering_error == 2:
        state = net.state_dict()
        clustering_error.append(state2clustering_error(state, args))

    out = {
        "args": args,
        "epoch": epochs_list,
        "train loss": loss,
        "terr": terr,
        "locality": locality,
        "stability": stability,
        "clustering_error": clustering_error,
        "dynamics": dynamics,
        "init": cpu_state_dict(net0) if args.save_init_net else None,
        "best": best,
        "last": cpu_state_dict(net) if args.save_last_net else None,
        "weight_evo": wo,
        'avg_epoch_time': avg_epoch_time,
    }
    yield out


def train(args, trainloader, net0, criterion):

    net = copy.deepcopy(net0)
    optimizer, scheduler = opt_algo(args, net)
    print(f"Training for {args.epochs} epochs...")

    start_time = time.time()

    num_batches = math.ceil(args.ptr / args.batch_size)
    checkpoint_batches = torch.linspace(0, num_batches, 10, dtype=int)

    for epoch in range(args.epochs):

        # layerwise training
        if epoch % (args.epochs // args.net_layers + 1) == 0:
            if 'layerwise' in args.net:
                l = epoch // (args.epochs // args.net_layers + 1)
                net.init_layerwise_(l)
                print(f'Layer-wise training up to layer {l}.', flush=True)

        net.train()
        train_loss = 0
        correct = 0
        total = 0
        total_batch = 0
        path_shallow_list = []
        path_deep_list = []
        target_list = []
        for batch_idx, (inputs, targets, path_shallow, path_deep) in enumerate(trainloader):
            inputs, targets, path_shallow, path_deep = inputs.to(args.device), targets.to(args.device), path_shallow.to(args.device), path_deep.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)

            path_shallow = F.one_hot(path_shallow, num_classes = 8)
            path_shallow_list.append(path_shallow)

            path_deep = F.one_hot(path_deep, num_classes = 8)
            path_deep_list.append(path_deep)

            target = F.one_hot(targets, num_classes = 8)
            target_list.append(target)
            total_batch += 1
            
            loss = criterion(outputs, targets)
            train_loss += loss.detach().item()
            regularize(loss, net, args.weight_decay, reg_type=args.reg_type)
            loss.backward()
            optimizer.step()

            correct, total = measure_accuracy(args, outputs, targets, correct, total)
            # during first epoch, save some sgd steps instead of after whole epoch
            if epoch < 10 and batch_idx in checkpoint_batches and batch_idx != (num_batches - 1):
                yield net, epoch + (batch_idx + 1) / num_batches, train_loss / (batch_idx + 1), None

        avg_epoch_time = (time.time() - start_time) / (epoch + 1)

        if (epoch % 1 == 0):
            features = net.feas
            features1 = []
            features2 = []
            features3 = []
            features4 = []

            for i in range(total_batch):         # collect the features in different batches
                features1.append(features[i*4+0])
                features2.append(features[i*4+1])
                features3.append(features[i*4+2])
                features4.append(features[i*4+3])

            total_batch = 0
            input = torch.cat(net.input, dim = 0) # concatenate the input, intermediate path, and target from different patches
            path_shallow = torch.cat(path_shallow_list, dim = 0) 
            path_deep = torch.cat(path_deep_list, dim = 0)
            target = torch.cat(target_list, dim = 0)
            
            v_1 = torch.cat(features1, dim = 0) 
            v_2 = torch.cat(features2, dim = 0)
            v_3 = torch.cat(features3, dim = 0)
            v_4 = torch.cat(features4, dim = 0)


            net.feas1_probe = v_1       # store the features in the train set, then we use them in test()
            net.feas2_probe = v_2
            net.feas3_probe = v_3
            net.feas4_probe = v_4
            net.input_probe = input     # store training data, then we use them in test() to learn a linear model
            net.label_probe = target

            inputs_dimension = input.size(0)     # inputs_dimension = number of samples
            net.input_dimension = inputs_dimension
            net.path_shallow_probe = path_shallow
            net.path_deep_probe = path_deep

            net.feas = [] # When recording the features, input data, intermediate path, and class label of this epoch, clear these containers and record the next epoch.              
            net.input = []
            path_shallow_list = []
            path_deep_list = []
            target_list = []

            print(
                f"[Train epoch {epoch+1} / {args.epochs}, {print_time(avg_epoch_time)}/epoch, ETA: {print_time(avg_epoch_time * (args.epochs - epoch - 1))}]"
                f"[tr.Loss: {train_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]",
                flush=True
            )

        scheduler.step()

        yield net, epoch + 1, train_loss / (batch_idx + 1), avg_epoch_time


def test(args, testloader, net, criterion, print_flag=True):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, path_shallow, path_deep) in enumerate(testloader):
            inputs, targets, path_shallow, path_deep = inputs.to(args.device), targets.to(args.device), path_shallow.to(args.device), path_deep.to(args.device)
            outputs = net(inputs)

            # Extract the features, input data, intermediate path, and class label stored in train(), and use them to learn a linear model.
            features1_probe = net.feas1_probe 
            features2_probe = net.feas2_probe
            features3_probe = net.feas3_probe
            features4_probe = net.feas4_probe
            input_probe = net.input_probe
            label_probe = net.label_probe
            path_shallow_probe = net.path_shallow_probe
            path_deep_probe = net.path_deep_probe


            input_probe_dimension = net.input_dimension
            label_probe_dimension = net.input_dimension

            # Refresh
            net.features1_probe = []
            net.features2_probe = []
            net.features3_probe = []
            net.features4_probe = []
            net.input_probe = []
            net.label_probe = []
            net.path_shallow_probe = []
            net.path_deep_probe = []
           
            
            
            v_1 = features1_probe
            v_2 = features2_probe
            v_3 = features3_probe
            v_4 = features4_probe

            v_1 = v_1.reshape(input_probe_dimension, -1)
            v_1 = v_1.detach()  
            v_1 = numpy.array(v_1)

            v_2 = v_2.reshape(input_probe_dimension, -1)
            v_2 = v_2.detach()  
            v_2 = numpy.array(v_2)

            v_3 = v_3.reshape(input_probe_dimension, -1)
            v_3 = v_3.detach()  
            v_3 = numpy.array(v_3)

            v_4 = v_4.reshape(input_probe_dimension, -1)
            v_4 = v_4.detach()  
            v_4 = numpy.array(v_4)


            u_4 = numpy.array(input_probe)
            u_4 = u_4.reshape(input_probe_dimension, -1) 

            u_3 = numpy.array(path_deep_probe)
            u_3 = u_3.reshape(input_probe_dimension, -1) 
            
            u_2 = numpy.array(path_shallow_probe)
            u_2 = u_2.reshape(input_probe_dimension, -1)

            u_1 = numpy.array(label_probe)
            u_1 = u_1.reshape(label_probe_dimension, -1)

            #print("New label_probe of training sample 1:", label_probe[0])
            #print("New label_probe of training sample 2:", label_probe[1])
            #print("New label_probe of training sample 3:", label_probe[2])
            #print("New label_probe of training sample 817:", label_probe[816])
            #print("New label_probe of training sample 818:", label_probe[817])
            #print("New label_probe of training sample 819:", label_probe[818])

            linear_model1 = LinearRegression()
            linear_model2 = LinearRegression()
            linear_model3 = LinearRegression()
            linear_model4 = LinearRegression()
            linear_model5 = LinearRegression()
            linear_model6 = LinearRegression()
            linear_model7 = LinearRegression()
            # learn a linear model using the features, input data, intermediate path, and class label stored in train().
            linear_model1.fit(v_1, u_4)  
            linear_model2.fit(v_2, u_4)  
            linear_model3.fit(v_3, u_4)  
            linear_model4.fit(v_4, u_4)  
            linear_model5.fit(u_1, u_4)  
            linear_model6.fit(u_2, u_4)  
            linear_model7.fit(u_3, u_4)  

            features = net.feas
            input = numpy.array(net.input)
            input_dimension = input.shape[1]
            #print("input_dimension", input.shape[1])
            
            net.input = []
            net.feas = []


            # Now these features, input data, intermediate path, and class label are from the test set, and the linear model is only used for prediction, not for updating.
            
            v_1 = features[0]
            v_2 = features[1]
            v_3 = features[2]
            v_4 = features[3]

            v_1 = v_1.reshape(input_dimension, -1)
            v_1 = v_1.detach()  
            v_1 = numpy.array(v_1)

            v_2 = v_2.reshape(input_dimension, -1)
            v_2 = v_2.detach()  
            v_2 = numpy.array(v_2)

            v_3 = v_3.reshape(input_dimension, -1)
            v_3 = v_3.detach()  
            v_3 = numpy.array(v_3)

            v_4 = v_4.reshape(input_dimension, -1)
            v_4 = v_4.detach()  
            v_4 = numpy.array(v_4)


            u_4 = numpy.array(input)              
            u_4 = u_4.reshape(input_dimension, -1)

            #print("input", u_4[0])    # testset
            #print("input", u_4[1])    # testset
            #print("input", u_4[2])    # testset
            #print("input", u_4[201])    # testset
            #print("input", u_4[202])    # testset
            #print("input", u_4[203])    # testset

            u_3 = F.one_hot(path_deep, num_classes = 8)
            u_3 = numpy.array(u_3)
            u_3 = u_3.reshape(input_dimension, -1)

            u_2 = F.one_hot(path_shallow, num_classes = 8)
            u_2 = numpy.array(u_2)
            u_2 = u_2.reshape(input_dimension, -1)

            u_1 = F.one_hot(targets, num_classes = 8)
            u_1 = numpy.array(u_1)
            u_1 = u_1.reshape(input_dimension, -1)
            
            #print("label of testing sample 1:", u_1[0])
            #print("label of testing sample 2:", u_1[1])
            #print("label of testing sample 3:", u_1[2])
            #print("label of testing sample 202:", u_1[201])
            #print("label of testing sample 203:", u_1[202])
            #print("label of testing sample 204:", u_1[203])
            
            # Predict using the trained model and features, input data, intermediate path, and class label from the test set
            predictions1 = linear_model1.predict(v_1)  
            predictions2 = linear_model2.predict(v_2)  
            predictions3 = linear_model3.predict(v_3)  
            predictions4 = linear_model4.predict(v_4)  
            predictions5 = linear_model5.predict(u_1)  
            predictions6 = linear_model6.predict(u_2)  
            predictions7 = linear_model7.predict(u_3)

            # When predicting different quantities, the parameters of the linear model need to change. 
            # For example, u_4 here represents predicting input data. 
            # To predict the intermediate path/hidden feature, replace u_4 with u_2 or u_3.
            # Accordingly, the parameters and title in plot() also need to be modified.

            mse1 = mean_squared_error(u_4, predictions1)  
            mse2 = mean_squared_error(u_4, predictions2)  
            mse3 = mean_squared_error(u_4, predictions3)  
            mse4 = mean_squared_error(u_4, predictions4)  
            mse5 = mean_squared_error(u_4, predictions5)  
            mse6 = mean_squared_error(u_4, predictions6)  
            mse7 = mean_squared_error(u_4, predictions7)

            net.mse_test1.append(mse1)
            net.mse_test2.append(mse2)
            net.mse_test3.append(mse3)
            net.mse_test4.append(mse4)
            net.mse_test5.append(mse5)
            net.mse_test6.append(mse6)
            net.mse_test7.append(mse7)


            loss = criterion(outputs, targets)

            test_loss += loss.item()

            correct, total = measure_accuracy(args, outputs, targets, correct, total)

        if print_flag:
            print(
                f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]",
                f"[MSE1: {mse1}]",
                f"[MSE2: {mse2}]",
                f"[MSE3: {mse3}]",
                f"[MSE4: {mse4}]",
                f"[MSE5: {mse5}]",
                f"[MSE6: {mse6}]",
                f"[MSE7: {mse7}]",
                flush=True
            )
            time_points = list(range(1, len(net.mse_test1) + 1))

        
        if len(net.mse_test1) == 25:     # last test epoch
        # When changing the predict object, please modify the plot() function at the top of this program and the title here. 
            plot(time_points, net.mse_test1, net.mse_test2, net.mse_test3, net.mse_test4, net.mse_test5, net.mse_test6, net.mse_test7, 'MSE on Probing Original data input of RHM (FCN, Test time)')


    return 100.0 * correct / total


# timing function
def print_time(elapsed_time):

    # if less than a second, print milliseconds
    if elapsed_time < 1:
        return f"{elapsed_time * 1000:.00f}ms"

    elapsed_seconds = round(elapsed_time)

    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)

    elapsed_time = []
    if h > 0:
        elapsed_time.append(f"{h}h")
    if not (h == 0 and m == 0):
        elapsed_time.append(f"{m:02}m")
    elapsed_time.append(f"{s:02}s")

    return "".join(elapsed_time)


def weights_evolution(f0, f):
    s0 = f0.state_dict()
    s = f.state_dict()
    nd = 0
    for k in s:
        nd += (s0[k] - s[k]).norm() / s0[k].norm()
    nd /= len(s)
    return nd


def main():

    parser = argparse.ArgumentParser()

    ### Tensors type ###
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")

    ### Seeds ###
    parser.add_argument("--seed_init", type=int, default=0)  # seed random-hierarchy-model
    parser.add_argument("--seed_net", type=int, default=-1)  # network initalisation
    parser.add_argument("--seed_trainset", type=int, default=-1)  # training sample

    ### DATASET ARGS ###
    parser.add_argument("--dataset", type=str, required=True)    # hier1 for hierarchical
    parser.add_argument("--ptr", type=float, default=0.8,
        help="Number of training point. If in [0, 1], fraction of training points w.r.t. total. If negative argument, P = |arg|*P_star",
    )
    parser.add_argument("--pte", type=float, default=.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scale_batch_size", type=int, default=0)

    parser.add_argument("--background_noise", type=float, default=0)

    # Hierarchical dataset #
    parser.add_argument("--num_features", type=int, default=8)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--input_format", type=str, default="onehot")
    parser.add_argument("--whitening", type=int, default=0)
    parser.add_argument("--auto_regression", type=int, default=0)    # not for now

    ### ARCHITECTURES ARGS ###
    parser.add_argument("--net", type=str, required=True)    # fcn or cnn
    parser.add_argument("--random_features", type=int, default=0)

    ## Nets params ##
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--net_layers", type=int, default=3)
    parser.add_argument("--filter_size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument("--bias", type=int, default=1, help="for some archs, controls bias presence")

    ## Auto-regression with Transformers ##
    parser.add_argument("--pmask", type=float, default=.2)    # not for now


    ### ALGORITHM ARGS ###
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--scheduler", type=str, default="cosineannealing")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--reg_type", default='l2', type=str)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--zero_loss_epochs", type=int, default=0)
    parser.add_argument("--zero_loss_threshold", type=float, default=0.01)
    parser.add_argument("--rescale_epochs", type=int, default=0)

    parser.add_argument(
        "--alpha", default=1.0, type=float, help="alpha-trick parameter"
    )

    ### Observables ###
    # how to use: 1 to compute stability every checkpoint; 2 at end of training. Default 0.
    parser.add_argument("--stability", type=int, default=0)
    parser.add_argument("--clustering_error", type=int, default=0)
    parser.add_argument("--locality", type=int, default=0)

    ### SAVING ARGS ###
    parser.add_argument("--save_init_net", type=int, default=1)
    parser.add_argument("--save_best_net", type=int, default=1)
    parser.add_argument("--save_last_net", type=int, default=1)
    parser.add_argument("--save_dynamics", type=int, default=0)

    ## saving path ##
    parser.add_argument("--pickle", type=str, required=False, default="None")
    parser.add_argument("--output", type=str, required=False, default="None")
    args = parser.parse_args()

    if args.pickle == "None":
        assert (
            args.output != "None"
        ), "either `pickle` or `output` must be given to the parser!!"
        args.pickle = args.output

    # special value -1 to set some equal arguments
    if args.seed_trainset == -1:
        args.seed_trainset = args.seed_init
    if args.seed_net == -1:
        args.seed_net = args.seed_init
    if args.num_classes == -1:
        args.num_classes = args.num_features
    if args.net_layers == -1:
        args.net_layers = args.num_layers
    if args.m == -1:
        args.m = args.num_features

    # define train and test sets sizes

    args.ptr, args.pte = args2train_test_sizes(args)

    with open(args.output, "wb") as handle:
        pickle.dump(args, handle)
    try:
        for data in run(args):
            with open(args.output, "wb") as handle:
                pickle.dump(args, handle)
                pickle.dump(data, handle)
    except:
        os.remove(args.output)
        raise


if __name__ == "__main__":
    main()
