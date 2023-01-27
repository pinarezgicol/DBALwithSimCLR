import argparse
from acquisition_functions import *
from models import load_pretrained_model, load_model
from utils import *
from al import *
import numpy as np
from skorch import NeuralNetClassifier
import time

if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Active Learning Setting')

    parser.add_argument('--network_type', type=str, default="Self-Supervised")

    parser.add_argument('--resnet', type=str, default="resnet18")

    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--results_folder', type=str)

    parser.add_argument('--dataset', type=str, default="CIFAR10")

    parser.add_argument('--projection_dim', type=int, default=64)

    parser.add_argument('--max_epochs', type=int, default=200)

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--experiment_count', type=int, default=3)

    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    ACQ_FUNCS = {
        "bald": bald,
        "var_ratios": var_ratios,
        "mean_std": mean_std,
        "max_entropy": max_entropy,
        "uniform": uniform
    }

    train_dataset, test_dataset = load_data(dataset=args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))

    if args.network_type == "Self-Supervised":
        nn_model = load_pretrained_model(args.resnet, args.model_path)
    else:
        nn_model = load_model(args.resnet)

    number_of_classes = 10

    for exp_iter in range(args.experiment_count):
        np.random.seed(exp_iter)
        initial_idx = np.array([], dtype=int)
        for i in np.unique(y_train).astype(int):
            idx = np.random.choice(np.where(y_train == i)[0], size=2, replace=False)
            initial_idx = np.concatenate((initial_idx, idx))

        for func_name, acquisition_func in ACQ_FUNCS.items():
            X_initial = X_train[initial_idx]
            y_initial = np.asarray(y_train)[initial_idx]

            X_pool = np.delete(X_train, initial_idx, axis=0)
            y_pool = np.delete(y_train, initial_idx, axis=0)

            estimator = NeuralNetClassifier(nn_model,
                                            max_epochs=args.max_epochs,
                                            batch_size=args.batch_size,
                                            lr=args.learning_rate,
                                            optimizer=torch.optim.Adam,
                                            criterion=torch.nn.CrossEntropyLoss,
                                            train_split=None,
                                            verbose=0,
                                            device=DEVICE)

            file_name = os.path.join(args.results_folder, func_name + "_exp_" + str(exp_iter) + ".npy")
            start = time.time()
            acc_arr, dataset_size_arr, y_queried_labels = active_learning_procedure(acquisition_func,
                                                                                    X_test,
                                                                                    y_test,
                                                                                    X_pool,
                                                                                    y_pool,
                                                                                    X_initial,
                                                                                    y_initial,
                                                                                    estimator,
                                                                                    file_name,
                                                                                    n_instances=number_of_classes)

            end = time.time()
            print("Time Elapsed: ", end-start)
