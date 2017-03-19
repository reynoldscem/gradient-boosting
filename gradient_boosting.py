from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from theano import tensor as T
import numpy as np
import theano
import scipy


def get_data():
    diabetes = load_diabetes()
    X, y = (diabetes[key] for key in ('data', 'target'))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.10
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.10
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_baseline(X, y):
    tuned_params = {
        'criterion': ['mse', 'mae'],
        'splitter': ['best', 'random'],
        # 'max_features': np.arange(1, 11),
        'max_depth': np.arange(1, 16),
        # 'min_samples_split': np.arange(2, 16),
        # 'min_samples_leaf': np.arange(1, 16)
    }
    clf = GridSearchCV(
        DecisionTreeRegressor(),
        tuned_params,
        cv=5,
        n_jobs=4
    )
    clf.fit(X, y)
    return clf.best_params_, clf


def get_error_and_res_functions():
    y_true = T.vector()
    y_pred = T.vector()
    mse = ((y_true - y_pred) ** 2).mean()
    mse_fun = theano.function(
        [y_pred, y_true],
        mse
    )
    residuals = -T.grad(mse, y_pred)
    res_fun = theano.function(
        [y_pred, y_true],
        residuals
    )
    return mse_fun, res_fun


def ens_predict(X_train, models, coeffs):
    pred_all = [
        models[i](X_train) * coeffs[i]
        for i in range(len(models))
    ]
    return np.sum(np.vstack(pred_all), axis=0)


def main():
    max_rounds = 200
    eps = 1e-2
    v = 0.05

    X_train, X_val, X_test, y_train, y_val, y_test = get_data()

    weak_learner_params, baseline_clf = get_baseline(
        X_train, y_train
    )

    baseline_pred_train = baseline_clf.predict(X_train)
    baseline_pred_val = baseline_clf.predict(X_val)

    mse_fun, res_fun = get_error_and_res_functions()
    print(
        'Train MSE for baseline {:.2f}'.format(
            float(mse_fun(baseline_pred_train, y_train))
        )
    )
    print(
        'Val MSE for baseline {:.2f}'.format(
            float(mse_fun(baseline_pred_val, y_val))
        )
    )

    def f_zero(x, gamma=np.mean(y_train)):
        return np.ones(x.shape[0]) * gamma

    models = [f_zero]
    coeffs = [1.]
    pred_ens_train = ens_predict(X_train, models, coeffs)
    pred_ens_val = ens_predict(X_val, models, coeffs)
    train_mse_for_round = float(mse_fun(pred_ens_train, y_train))
    val_mse_for_round = float(mse_fun(pred_ens_val, y_val))
    print(
        'Train MSE for boosting round #{} {:.2f}'.format(
            0,
            train_mse_for_round
        )
    )
    print(
        'Val for boosting round #{} {:.2f}'.format(
            0,
            val_mse_for_round
        )
    )

    train_errors = [train_mse_for_round]
    val_errors = [val_mse_for_round]

    for boosting_round in range(1, max_rounds + 1):
        res_vals = res_fun(pred_ens_train, y_train)

        new_model = DecisionTreeRegressor(
            **weak_learner_params
        ).fit(X_train, res_vals)

        pred_ens = ens_predict(X_train, models, coeffs)
        new_gamma = scipy.optimize.brent(
            lambda coeff: mse_fun(
                y_train,
                pred_ens + coeff * new_model.predict(X_train)
            )
        )

        models.append(new_model.predict)
        coeffs.append(new_gamma * v)

        # Get the ensemble prediction.
        pred_ens_train = ens_predict(X_train, models, coeffs)
        pred_ens_val = ens_predict(X_val, models, coeffs)
        train_mse_for_round = float(mse_fun(pred_ens_train, y_train))
        val_mse_for_round = float(mse_fun(pred_ens_val, y_val))
        print(
            'Train MSE for boosting round #{} {:.2f}'.format(
                boosting_round,
                train_mse_for_round
            )
        )
        print(
            'Val MSE for boosting round #{} {:.2f}'.format(
                boosting_round,
                val_mse_for_round
            )
        )
        train_errors.append(train_mse_for_round)
        val_errors.append(val_mse_for_round)
        if train_mse_for_round < eps:
            print('Fit training data.')
            break

    from matplotlib import pyplot as plt
    # plt.figure()
    plt.plot(list(range(0, len(train_errors))), train_errors, label='train')
    plt.plot(list(range(0, len(val_errors))), val_errors, label='val')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
