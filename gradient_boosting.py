from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from theano import tensor as T
import numpy as np
import theano
import scipy


def main():
    diabetes = load_diabetes()
    X, y = (diabetes[key] for key in ('data', 'target'))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.10
    )

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
    clf.fit(X_train, y_train)
    weak_learner_params = clf.best_params_

    y_true = T.vector()
    y_pred = T.vector()
    mse = ((y_true - y_pred) ** 2).mean()
    mse_fun = theano.function(
        [y_pred, y_true],
        mse
    )
    print(
        'MSE for baseline {:.2f}'.format(
            float(mse_fun(clf.predict(X_train), y_train))
        )
    )

    def f_zero(x, gamma=np.mean(y_train)):
        return np.ones(x.shape[0]) * gamma

    models = [f_zero]
    coeffs = [1.]
    pred_all = [
        models[i](X_train) * coeffs[i]
        for i in range(len(models))
    ]
    pred_ens = np.sum(np.vstack(pred_all), axis=0)
    print(
        'MSE for 0-th boosting round {:.2f}'.format(
            float(mse_fun(pred_ens, y_train))
        )
    )

    residuals = -T.grad(mse, y_pred)
    res_fun = theano.function(
        [y_pred, y_true],
        residuals
    )

    # Adding one model.
    res_vals = res_fun(f_zero(X_train), y_train)

    f_1 = DecisionTreeRegressor(
        **weak_learner_params
    ).fit(X_train, res_vals)

    new_gamma = scipy.optimize.brent(
        lambda coeff: mse_fun(
            y_train,
            f_zero(X_train) + coeff * f_1.predict(X_train)
        )
    )

    models.append(f_1.predict)
    coeffs.append(new_gamma)

    # Get the ensemble prediction.
    pred_all = [
        models[i](X_train) * coeffs[i]
        for i in range(len(models))
    ]
    pred_ens = np.sum(np.vstack(pred_all), axis=0)
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
    pred_all = [
        models[i](X_train) * coeffs[i]
        for i in range(len(models))
    ]
    pred_ens = np.sum(np.vstack(pred_all), axis=0)
    print(
        'MSE for 2nd boosting round {:.2f}'.format(
            float(mse_fun(pred_ens, y_train))
        )
    )
    print(coeffs)


if __name__ == '__main__':
    main()
