import numpy as np
import matplotlib.pyplot as plt
import math


# Defining the 3/8 Method
def ThreeEightsMethod(f, x0, t0, tf, dt):
    a21 = 1. / 3.
    a31 = -1. / 3.
    a32 = 1.
    a41 = 1.
    a42 = -1.
    a43 = 1.

    b1 = 1. / 8.
    b2 = 3. / 8.
    b3 = 3. / 8.
    b4 = 1. / 8.

    c2 = 1. / 3.
    c3 = 2. / 3.
    c4 = 1.

    t = np.arange(t0, tf, dt)
    nt = t.size
    nx = x0.size
    x = np.zeros((nx, nt))
    x[:, 0] = x0
    error = 0
    for k in range(nt - 1):
        k1 = dt * f(t[k], x[:, k])
        k2 = dt * f(t[k] + dt * a21, x[:, k] + k1 * a21)
        k3 = dt * f(t[k] + dt * (a31 + a32), x[:, k] + k1 * a31 + k2 * a32)
        k4 = dt * f(t[k] + dt * (a41 + a42 + a43), x[:, k] + k1 * a41 + k2 * a42 + k3 * a43)
        dx = b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4
        x[:, k + 1] = x[:, k] + dx
        error += abs(x.max() ** 4 * dt ** 5 * max(k1) * 9.91 / 100)
    return x, t, error


def compare(arr1, arr2):
    for k in range(len(arr1)):
        if arr1[k] > arr2[k]:
            return 1
    return 0


# Defining the Dormand Prince Method
def DormandPrince(f, x0, t0, atol, rtol, h, maxiter):
    a21 = (1.0 / 5.0)
    a31 = (3.0 / 40.0)
    a32 = (9.0 / 40.0)
    a41 = (44.0 / 45.0)
    a42 = (-56.0 / 15.0)
    a43 = (32.0 / 9.0)
    a51 = (19372.0 / 6561.0)
    a52 = (-25360.0 / 2187.0)
    a53 = (64448.0 / 6561.0)
    a54 = (-212.0 / 729.0)
    a61 = (9017.0 / 3168.0)
    a62 = (-355.0 / 33.0)
    a63 = (46732.0 / 5247.0)
    a64 = (49.0 / 176.0)
    a65 = (-5103.0 / 18656.0)
    a71 = (35.0 / 384.0)
    a72 = (0.0)
    a73 = (500.0 / 1113.0)
    a74 = (125.0 / 192.0)
    a75 = (-2187.0 / 6784.0)
    a76 = (11.0 / 84.0)

    c2 = (1.0 / 5.0)
    c3 = (3.0 / 10.0)
    c4 = (4.0 / 5.0)
    c5 = (8.0 / 9.0)
    c6 = (1.0)
    c7 = (1.0)

    b1 = (35.0 / 384.0)
    b2 = (0.0)
    b3 = (500.0 / 1113.0)
    b4 = (125.0 / 192.0)
    b5 = (-2187.0 / 6784.0)
    b6 = (11.0 / 84.0)
    b7 = (0.0)

    b1p = (5179.0 / 57600.0)
    b2p = (0.0)
    b3p = (7571.0 / 16695.0)
    b4p = (393.0 / 640.0)
    b5p = (-92097.0 / 339200.0)
    b6p = (187.0 / 2100.0)
    b7p = (1.0 / 40.0)

    order = x0.size
    t = np.zeros(maxiter + 1)
    t[0] = t0
    x = np.zeros((order, maxiter + 1))
    x[:, 0] = x0
    error = 0
    for i in range(maxiter):
        x0 = x[:, i]
        K1 = h * f(t[i], x0)
        K2 = h * f(t[i] + c2 * h, x[:, i] + (a21 * K1))
        K3 = h * f(t[i] + c3 * h, x[:, i] + (a31 * K1 + a32 * K2))
        K4 = h * f(t[i] + c4 * h, x[:, i] + (a41 * K1 + a42 * K2 + a43 * K3))
        K5 = h * f(t[i] + c5 * h, x[:, i] + (a51 * K1 + a52 * K2 + a53 * K3 + a54 * K4))
        K6 = h * f(t[i] + c6 * h, x[:, i] + (a61 * K1 + a62 * K2 + a63 * K3 + a64 * K4 + a65 * K5))
        K7 = h * f(t[i] + c7 * h, x[:, i] + (a71 * K1 + a72 * K2 + a73 * K3 + a74 * K4 + a75 * K5 + a76 * K6))

        delta = abs((b1 - b1p) * K1 + (b3 - b3p) * K3 + (b4 - b4p) * K4
                    + (b5 - b5p) * K5 + (b6 - b6p) * K6 + (b7 - b7p) * K7)
        error += max(delta)
        tol = []
        x1 = x[:, i]
        for k in range(order):
            x2 = x0[k] + (b1p * K1[k] + b3p * K3[k] + b4p * K4[k] + b5p * K5[k] + b6p * K6[k])
            tol.append(atol + max(abs(x1[k]), abs(x2)) * rtol)

        if compare(delta, tol) == 1:  # change step
            sum = 0
            for k in range(order):
                sum += (delta[k] / tol[k]) ** 2

            err = math.sqrt(1 / order * sum)
            if sum != 0:
                h = h * (1 / err) ** (1 / 5)
            dx = 0
        else:
            dx = (b1 * K1 + b3 * K3 + b4 * K4 + b5 * K5 + b6 * K6)
        x[:, i + 1] = x[:, i] + dx
        t[i + 1] = t[i] + h

    error *= 10 ** -6
    return x, t, error


# Defining the Ralston 4 Method
def Ralston4(f, x0, t0, tf, dt):
    a21 = 0.4

    a31 = (-2889 + 1428 * math.sqrt(5)) / 1024
    a32 = (3875 - 1620 * math.sqrt(5)) / 1024

    a41 = (-3365 + 2094 * math.sqrt(5)) / 6040
    a42 = (-975 - 3046 * math.sqrt(5)) / 2552
    a43 = (467040 + 203968 * math.sqrt(5)) / 240845

    b1 = (263 + 24 * math.sqrt(5)) / 1812
    b2 = (125 - 1000 * math.sqrt(5)) / 3828
    b3 = 1024 * (3346 + 1623 * math.sqrt(5)) / 5924787
    b4 = (30 - 4 * math.sqrt(5)) / 123

    t = np.arange(t0, tf, dt)
    nt = t.size
    nx = x0.size
    x = np.zeros((nx, nt))
    x[:, 0] = x0
    error = 0
    for k in range(nt - 1):
        k1 = dt * f(t[k], x[:, k])
        k2 = dt * f(t[k] + dt * a21, x[:, k] + k1 * a21)
        k3 = dt * f(t[k] + dt * (a31 + a32), x[:, k] + k1 * a31 + k2 * a32)
        k4 = dt * f(t[k] + dt * (a41 + a42 + a43), x[:, k] + k1 * a41 + k2 * a42 + k3 * a43)
        dx = b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4
        x[:, k + 1] = x[:, k] + dx
        if k == 0:
            error += abs(x.max() ** 4 * dt ** 5 * max(k1) * 5.46 / 100)
    return x, t, error


# Defining the Runge Kutta 4 Method
def RK4(f, x0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    nt = t.size
    nx = x0.size
    x = np.zeros((nx, nt))
    x[:, 0] = x0
    error = 0
    for k in range(nt - 1):
        k1 = dt * f(t[k], x[:, k])
        k2 = dt * f(t[k] + dt / 2, x[:, k] + k1 / 2)
        k3 = dt * f(t[k] + dt / 2, x[:, k] + k2 / 2)
        k4 = dt * f(t[k] + dt, x[:, k] + k3)
        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[:, k + 1] = x[:, k] + dx
        error += abs(x.max() ** 4 * dt ** 5 * max(k1) * 10.14 / 100)
    return x, t, error


def print_graphics(t, x, labels):
    plt.subplot(1, 2, 1)
    plt.plot(t, x[0, :], "r", label=labels[0])
    plt.plot(t, x[1, :], "b", label=labels[1])
    plt.ylabel("Количество (тыс.)")
    plt.xlabel("Время (t)")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x[0, :], x[1, :])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.grid()
    plt.show()


def print_graphic(t, x, labels):
    plt.plot(t, x[0, :], "r", label=labels[0])
    plt.plot(t, x[1, :], "b", label=labels[1])
    plt.plot(t, x[2, :], "g", label=labels[2])
    plt.ylabel("Количество (тыс.)")
    plt.xlabel("Время (t)")
    plt.grid()
    plt.legend()
    plt.show()


def Rabbits_Foxes_Function(x, params):
    e1 = params['e1']
    e2 = params['e2']
    a1 = params['a1']
    a2 = params['a2']
    k1 = params['k1']
    new_x = np.array([x[0] * (e1 - a1 * x[1]) * (1 - x[0] / k1),
                      -x[1] * (e2 - a2 * x[0])])
    return new_x


def Rabbits_Foxes():
    params = {"e1": 4, "a1": 2, "k1": 5,
              "e2": 2, "a2": 1}

    f = lambda t, x: Rabbits_Foxes_Function(x, params)
    x0 = np.array([2., 1.])  # initial condition
    t0 = 0  # time
    tf = 10  # end of time
    dt = 0.01  # step
    atol = 1.0e-5
    rtol = 1
    maxiter = int(tf / dt)
    # x, t, error = DormandPrince(f, x0, t0, atol, rtol, dt, maxiter)
    # x, t, error = RK4(f, x0, t0, tf, dt)
    # x, t, error = ThreeEightsMethod(f, x0, t0, tf, dt)
    # x, t, error = Ralston4(f, x0, t0, tf, dt)
    print(error)
    labels = ["Rabbits", "Foxes"]
    print_graphics(t, x, labels)


# def Producer_Consumer_Predator_Function(x, params):
#     a = params['a']
#     l = params['l']
#     k = params['k']
#     g1 = params['g1']
#     g2 = params['g2']
#     h1 = params['h1']
#     h2 = params['h2']
#     c1 = params['c1']
#     c2 = params['c2']
#     new_x = np.array([a * x[0] * (x[0] - l) * (1 - x[0] / k) - g1 * x[0] * x[1],
#                       h1 * x[0] * x[1] - g2 * x[0] * x[1] - c1 * x[1],
#                       h2 * x[1] * x[2] - c2 * x[2]])
#     return new_x


# def Producer_Consumer_Predator():
#     params = {"a": 30, "l": 1, "k": 60,
#               "g1": 3, "g2": 1, "h1": 4,
#               "h2": 2, "c1": 1, "c2": 2}
#     f = lambda t, x: Producer_Consumer_Predator_Function(x, params)
#     x0 = np.array([10., 3., 1.])
#     t0 = 0
#     tf = 10
#     dt = 0.01
#     x, t, error = RK4(f, x0, t0, tf, dt)
#     print(error)
#     labels = ["Продуцент", "Консумент", "Хищник"]
#     print_graphic(t, x, labels)


def Prey_Prey_Predator_Function(x, params):
    alpha = params['alpha']
    beta = params['beta']
    a1 = params['a1']
    a2 = params['a2']
    b1 = params['b1']
    b2 = params['b2']
    d1 = params['d1']
    d2 = params['d2']
    new_x = np.array([x[0] * (alpha - x[0] - a1 * x[1] - b1 * x[2]),
                      x[1] * (beta - x[1] - a2 * x[0] - b2 * x[2]),
                      x[2] * (-1 + d1 * x[0] + d2 * x[1] - x[2])])
    return new_x


def Prey_Prey_Predator():
    params = {"alpha": 6, "a1": 5, "b1": 4.,
              "beta": 9, "a2": 1, "b2": 10.,
              "d1": 0.25, "d2": 4}

    f = lambda t, x: Prey_Prey_Predator_Function(x, params)
    x0 = np.array([3, 1, 2])  # initial condition
    t0 = 0  # time
    tf = 10  # end of time
    dt = 0.01  # step
    atol = 1.0e-5
    rtol = 1
    maxiter = int(tf / dt)
    x, t, error = DormandPrince(f, x0, t0, atol, rtol, dt, maxiter)
    # x, t, error = RK4(f, x0, t0, tf, dt)
    # x, t, error = ThreeEightsMethod(f, x0, t0, tf, dt)
    # x, t, error = Ralston4(f, x0, t0, tf, dt)
    print(error)
    labels = ["Prey1", "Prey2", "Predator"]
    print_graphic(t, x, labels)


def Rabbits_Foxes_Dissipative_Function(x, params):
    e1 = params['e1']
    e2 = params['e2']
    a1 = params['a1']
    a2 = params['a2']
    a3 = params['a3']
    a4 = params['a4']
    new_x = np.array([x[0] * (e1 - a1 * x[0] - a2 * x[1]),
                      x[1] * (e2 + a3 * x[0] - a4 * x[1])])
    return new_x


def Rabbits_Foxes_Dissipative():  # Модель консумента 1-ого порядка и консумента 2-ого порядка
    params = {"e1": 4, "a1": 2, "a2": 2,
              "e2": 2, "a3": 1, "a4": 8}

    f = lambda t, x: Rabbits_Foxes_Dissipative_Function(x, params)
    x0 = np.array([3., 5.])  # initial condition
    t0 = 0  # time
    tf = 10  # end of time
    dt = 0.01  # step
    atol = 1.0e-5
    rtol = 1
    maxiter = int(tf / dt)
    x, t, error = DormandPrince(f, x0, t0, atol, rtol, dt, maxiter)
    # x, t, error = RK4(f, x0, t0, tf, dt)
    # x, t, error = ThreeEightsMethod(f, x0, t0, tf, dt)
    # x, t, error = Ralston4(f, x0, t0, tf, dt)
    print(error)
    # x, t, error = RK4(f, x0, t0, tf, dt)
    labels = ["Rabbits", "Foxes"]
    print_graphics(t, x, labels)


def Sheep_Rabbits_Function(x, params):
    e1 = params['e1']
    e2 = params['e2']
    a1 = params['a1']
    a2 = params['a2']
    a3 = params['a3']
    a4 = params['a4']
    new_x = np.array([x[0] * (e1 + a1 * x[0] + a2 * x[1]),
                      x[1] * (e2 + a3 * x[0] + a4 * x[1])])
    return new_x


def Sheep_Rabbits():  # Конкуренция видов
    params = {"e1": 6, "a1": -2, "a2": -3,
              "e2": 8, "a3": -2, "a4": -4}

    f = lambda t, x: Sheep_Rabbits_Function(x, params)
    x0 = np.array([4., 6.])  # initial condition
    t0 = 0  # time
    tf = 10  # end of time
    dt = 0.01  # step
    atol = 1.0e-5
    rtol = 1
    maxiter = int(tf / dt)
    x, t, error = DormandPrince(f, x0, t0, atol, rtol, dt, maxiter)
    # x, t, error = RK4(f, x0, t0, tf, dt)
    # x, t, error = ThreeEightsMethod(f, x0, t0, tf, dt)
    # x, t, error = Ralston4(f, x0, t0, tf, dt)
    print(error)
    labels = ["Rabbits", "Sheep"]
    print_graphics(t, x, labels)

# Rabbits_Foxes()
# Rabbits_Foxes_Dissipative()
# Prey_Prey_Predator()
# Sheep_Rabbits()
