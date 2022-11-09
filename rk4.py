import numpy as np
import matplotlib.pyplot as plt


def LV(x, params):
    alpha = params['alpha']
    beta = params['beta']

    new_x = np.array([x[0] * (1 - x[0] - alpha * x[1] - beta * x[2]),
                      x[1] * (1 - beta * x[0] - x[1] - alpha * x[2]),
                      x[2] * (1 - alpha * x[0] - beta * x[1] - x[2])])
    # e1 = params['e1']
    # e2 = params['e2']
    # e3 = params['e3']
    # a1 = params['a1']
    # a2 = params['a2']
    # a3 = params['a3']
    # a4 = params['a4']
    # a5 = params['a5']
    # a6 = params['a6']
    # k1 = params['k1']
    # k3 = params['k2']
    # k2 = params['k3']
    # new_x = np.array([x[0] * (e1 - a1 * x[1] - k1 * x[0]),
    #                   -x[1] * (e2 - a3 * x[0] + k2 * x[1])])
    # new_x = np.array([x[0] * (e1 - a1 * x[1] - a2 * x[2]),
    #                  -x[1] * (e2 - a3 * x[0] - a4 * x[2]),
    #                  -x[2] * (e3 - a5 * x[0] - a6 * x[1])])
    # xdot = np.array([alpha * x[0] - beta * x[0] * x[1],
    #                  delta * x[0] * x[1] - gamma * x[1]])

    return new_x


# Defining the Runge Kutta 4 Method
def RK4(f, x0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    nt = t.size
    nx = x0.size
    x = np.zeros((nx, nt))
    x[:, 0] = x0
    for k in range(nt - 1):
        k1 = dt * f(t[k], x[:, k])
        k2 = dt * f(t[k] + dt / 2, x[:, k] + k1 / 2)
        k3 = dt * f(t[k] + dt / 2, x[:, k] + k2 / 2)
        k4 = dt * f(t[k] + dt, x[:, k] + k3)
        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[:, k + 1] = x[:, k] + dx
    return x, t


def print_graphics(t, x, num, labels):
    # plt.subplot(1, 2, 1)
    plt.plot(t, x[0, :], "r", label=labels[0])
    plt.plot(t, x[1, :], "b", label=labels[1])
    if num == 3:
        plt.plot(t, x[2, :], "g", label=labels[2])
    plt.ylabel("Количество (тыс.)")
    plt.xlabel("Время (t)")
    plt.grid()
    plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(x[0, :], x[1, :], x[2, :])
    # plt.xlabel("Preys")
    # plt.ylabel("Predators")
    # plt.grid()
    plt.show()


def Rabbits_Foxes_Function(x, params):
    e1 = params['e1']
    e2 = params['e2']
    a1 = params['a1']
    a2 = params['a2']
    new_x = np.array([x[0] * (e1 - a1 * x[1]),
                      -x[1] * (e2 - a2 * x[0])])
    return new_x


def Rabbits_Foxes():
    params = {"e1": 4, "a1": 2,
              "e2": 1, "a2": 1}

    f = lambda t, x: Rabbits_Foxes_Function(x, params)
    x0 = np.array([1., 1.])  # initial condition
    t0 = 0  # time
    tf = 30  # end of time
    dt = 0.01  # step
    x, t = RK4(f, x0, t0, tf, dt)
    labels = ["Rabbits", "Foxes"]
    print_graphics(t, x, 2, labels)


def Prey_Prey_Predator_Function(x, params):
    alpha = params['alpha']
    beta = params['beta']
    a1 = params['a1']
    a2 = params['a2']
    b1 = params['b1']
    b2 = params['b2']
    d1 = params['d1']
    d2 = params['d2']
    new_x = np.array([alpha - x[0] - a1 * x[1] - b1 * x[2],
                      beta - x[1] - a2 * x[0] - b2 * x[2],
                      x[2] * (-1 + d1 * x[0] + d2 * x[1] - x[2])])
    return new_x


def Producer_Consumer_Predator_Function(x, params):
    a = params['a']
    l = params['l']
    k = params['k']
    g1 = params['g1']
    g2 = params['g2']
    h1 = params['h1']
    h2 = params['h2']
    c1 = params['c1']
    c2 = params['c2']
    new_x = np.array([a * x[0] * (x[0] - l) * (1 - x[0] / k) - g1 * x[0] * x[1],
                      h1 * x[0] * x[1] - g2 * x[0] * x[1] - c1 * x[1],
                      h2 * x[1] * x[2] - c2 * x[2]])
    return new_x


def Producer_Consumer_Predator():
    params = {"a": 1, "l": 1, "k": 1000000,
              "g1": 1, "g2": 1, "h1": 2,
              "h2": 2, "c1": 1, "c2": 2}
    f = lambda t, x: Producer_Consumer_Predator_Function(x, params)
    x0 = np.array([1., 1., 1.])
    t0 = 0
    tf = 30
    dt = 0.01
    x, t = RK4(f, x0, t0, tf, dt)
    labels = ["Продуцент", "Консумент", "Хищник"]
    print_graphics(t, x, 3, labels)


def Prey_Prey_Predator():
    params = {"alpha": 2.4, "a1": 6, "b1": 4,
              "beta": 1.57, "a2": 1, "b2": 10,
              "d1": 0.25, "d2": 4}

    f = lambda t, x: Prey_Prey_Predator_Function(x, params)
    x0 = np.array([4, 3, 1])  # initial condition
    t0 = 0  # time
    tf = 1  # end of time
    dt = 0.01  # step
    x, t = RK4(f, x0, t0, tf, dt)
    labels = ["Prey1", "Prey2", "Predator"]
    print_graphics(t, x, 3, labels)


Producer_Consumer_Predator()
# Rabbits_Foxes()
#Prey_Prey_Predator()
