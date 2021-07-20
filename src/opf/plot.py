from matplotlib import pyplot as plt


def plot_equality(title, v1, v2):
    plt.figure()
    plt.plot(v1)
    plt.plot(v2)
    plt.legend(("Value 1", "Value 2"))
    plt.title(title)


def plot_inequality(title, value, lower, upper):
    plt.figure()
    plt.plot(value, "b")
    plt.plot(lower, "r")
    plt.plot(upper, "g")
    plt.legend(("val", "min", "max"))
    plt.title(title)
