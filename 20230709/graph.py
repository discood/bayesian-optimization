from bayesian import *
import matplotlib.pyplot as plt

#ガウス過程回帰のグラフ化

def plot_prediction(mean_prediction, std_prediction, X_observation, Y_observation):
    zeros = np.zeros(len(mean_prediction))
    plot_acq = np.maximum(ucb(mean_prediction,std_prediction,c)-7,zeros-7)
    max_value = max(plot_acq)
    max_index = np.argmax(plot_acq)
    xsin = function1(X_range)
    plt.plot(X_range, xsin, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.scatter(X_observation, Y_observation, label="Observations")
    plt.plot(X_range, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X_range.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.plot(X_range, plot_acq, label="acq", color = "green")
    plt.scatter(max_index*0.1,max_value, s=200, marker="*", label="param_next", color = "red")

    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f=xsin(x)$")
    plt.ylim(-7,5) #グラフ範囲を固定したい場合はここを調整
    #plt.title("iteration="+str(iteration+1))
    plt.savefig("graph/prediction.jpg")
    plt.show()

def plot_prediction_2(X):
    X = X.reshape(1,-1)
    func1 = function1(X)
    func2 = function2(X)
    iteration = np.linspace(0.1, 1, len(X[0]))
    plt.scatter(func1, func2, s=100, c=iteration, cmap='Blues')
    plt.xlabel("func1")
    plt.ylabel("func2")
    plt.colorbar()