import argparse
from copy import deepcopy
from pprint import pprint
from baal import ActiveLearningDataset
from baal.modelwrapper import ModelWrapper
from baal.active import get_heuristic, ActiveLearningLoop
from baal.bayesian.dropout import MCDropoutModule
from baal.utils.metrics import Accuracy
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from acquisition_functions import *
from models import load_pretrained_model, load_model
from utils import *


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Active Learning Setting')

    parser.add_argument('--self_supervised', type=bool, default=True)

    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--results_folder', type=str)

    parser.add_argument('--dataset', type=str, default="CIFAR10")

    parser.add_argument('--max_epochs', type=int, default=200)

    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--experiment_count', type=int, default=3)

    parser.add_argument('--freeze', type=bool, default=False)

    parser.add_argument('--dropout', type=bool, default=True)

    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    ACQ_FUNCS = ["bald", "random", "entropy"]

    for exp_iter in range(args.experiment_count):
        for acq_func in ACQ_FUNCS:
            train_dataset, test_dataset = load_data(dataset=args.dataset)

            active_set = ActiveLearningDataset(train_dataset, pool_specifics={"transform": TransformsSimCLR(size=train_dataset.data.shape[2]).test_transform})

            active_set.label_randomly(50)

            if args.self_supervised:
                nn_model = load_pretrained_model(args.model_path, freeze=args.freeze)
            else:
                nn_model = load_model(args.dropout)

            number_of_classes = 10

            heuristic = get_heuristic(acq_func, 0.05)

            criterion = CrossEntropyLoss()

            nn_model = MCDropoutModule(nn_model)

            if torch.cuda.is_available():
                nn_model.cuda()
            optimizer = optim.SGD(nn_model.parameters(), lr=args.learning_rate, momentum=0.9)

            nn_model = ModelWrapper(nn_model, criterion, replicate_in_memory=False)

            nn_model.add_metric(name='accuracy', initializer=lambda: Accuracy())

            logs = {}
            logs["epoch"] = 0

            active_loop = ActiveLearningLoop(
                active_set,
                nn_model.predict_on_dataset,
                heuristic,
                batch_size=args.batch_size,
                query_size=50,
                iterations=20,
                max_sample=5000,
                use_cuda=torch.cuda.is_available()
            )
            # We will reset the weights at each active learning step.
            init_weights = deepcopy(nn_model.state_dict())

            for _ in tqdm(range(19)):
                # Load the initial weights.
                nn_model.load_state_dict(init_weights)
                nn_model.train_on_dataset(
                    active_set,
                    optimizer,
                    args.batch_size,
                    args.max_epochs,
                    torch.cuda.is_available()
                )

                # Validation!
                nn_model.test_on_dataset(test_dataset, args.batch_size, torch.cuda.is_available())
                should_continue = active_loop.step()
                if not should_continue:
                    break

                pprint(nn_model.get_metrics())

                np.save(os.path.join(args.results_folder, getResultPath(args.dataset, args.self_supervised, acq_func, exp_iter, args.freeze)), nn_model.active_learning_metrics)