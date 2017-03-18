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

    return train_test_split(
        X, y,
        test_size=0.10
    )


def baseline_params_and_pred(X, y):
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
    return clf.best_params_, clf.predict(X)


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
    X_train, X_test, y_train, y_test = get_data()

    weak_learner_params, baseline_pred = baseline_params_and_pred(
        X_train, y_train
    )
    mse_fun, res_fun = get_error_and_res_functions()
    print(
        'MSE for baseline {:.2f}'.format(
            float(mse_fun(baseline_pred, y_train))
        )
    )

    def f_zero(x, gamma=np.mean(y_train)):
        return np.ones(x.shape[0]) * gamma

    models = [f_zero]
    coeffs = [1.]
    pred_ens = ens_predict(X_train, models, coeffs)
    print(
        'MSE for 0th boosting round {:.2f}'.format(
            float(mse_fun(pred_ens, y_train))
        )
    )

    # Adding one model.
    res_vals = res_fun(f_zero(X_train), y_train)

    f_1 = DecisionTreeRegressor(
        **weak_learner_params
    ).fit(X_train, res_vals)

    pred_ens = ens_predict(X_train, models, coeffs)
    new_gamma = scipy.optimize.brent(
        lambda coeff: mse_fun(
            y_train,
            pred_ens + coeff * f_1.predict(X_train)
        )
    )

    models.append(f_1.predict)
    coeffs.append(new_gamma)

    # Get the ensemble prediction.
    pred_ens = ens_predict(X_train, models, coeffs)
    print(
        'MSE for 1st boosting round {:.2f}'.format(
            float(mse_fun(pred_ens, y_train))
        )
    )

    # And the next.
    res_2 = res_fun(pred_ens, y_train)
    f_2 = DecisionTreeRegressor(
        **weak_learner_params
    ).fit(X_train, res_2)

    new_gamma = scipy.optimize.brent(
        lambda coeff: mse_fun(
            y_train,
            pred_ens + coeff * f_2.predict(X_train)
        )
    )

    models.append(f_2.predict)
    coeffs.append(new_gamma)
    pred_ens = ens_predict(X_train, models, coeffs)
    print(
        'MSE for 2nd boosting round {:.2f}'.format(
            float(mse_fun(pred_ens, y_train))
        )
    )

    # And the next.
    res_3 = res_fun(pred_ens, y_train)
    f_3 = DecisionTreeRegressor(
        **weak_learner_params
    ).fit(X_train, res_3)

    new_gamma = scipy.optimize.brent(
        lambda coeff: mse_fun(
            y_train,
            pred_ens + coeff * f_3.predict(X_train)
        )
    )

    models.append(f_3.predict)
    coeffs.append(new_gamma)
    pred_ens = ens_predict(X_train, models, coeffs)
    print(
        'MSE for 3rd boosting round {:.2f}'.format(
            float(mse_fun(pred_ens, y_train))
        )
    )


if __name__ == '__main__':
    main()
