import numpy as np
from modAL.models import ActiveLearner


def active_learning_procedure(query_strategy,
                              X_test,
                              y_test,
                              X_pool,
                              y_pool,
                              X_initial,
                              y_initial,
                              estimator,
                              file_name,
                              query_budget=1000,
                              n_instances=10):
    learner = ActiveLearner(estimator=estimator,
                            X_training=X_initial,
                            y_training=y_initial,
                            query_strategy=query_strategy
                            )
    perf_hist = [learner.score(X_test, y_test)]
    active_pool_size = [len(X_initial)]
    pool_size = len(X_initial)

    y_queried_labels = y_initial.tolist()

    index = 1
    while len(y_queried_labels) < query_budget:
        query_idx, query_instance = learner.query(X_pool, n_instances)

        y_queried_labels.extend(y_pool[query_idx])

        learner.teach(X_pool[query_idx], y_pool[query_idx])

        X_initial = np.append(X_initial, X_pool[query_idx], 0)

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        model_accuracy = learner.score(X_test, y_test)
        pool_size = pool_size + n_instances
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        index = index + 1
        perf_hist.append(model_accuracy)
        active_pool_size.append(pool_size)
        np.save(file_name, (perf_hist, active_pool_size, y_queried_labels))
    return perf_hist, active_pool_size, y_queried_labels


