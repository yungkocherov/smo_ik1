import numpy as np
import matplotlib.pyplot as plt


class StatMod:
    def __init__(self, n, requests_number, imitation_states, tmax=35, lamda=2, mu=1):
        self.lamda = lamda  # интенсивность появления новых заявок
        self.mu = mu  # интенсивность обработки заявки
        self.n = n  # число каналов обработки
        self.requests_number = requests_number  # общее число поступивших заявок (максимально возможное число состояний)
        self.max_states = imitation_states
        self.tmax = tmax  # максимально допустимый момент времени
        self.ts = []
        self.ys = []
        self.y0, self.t0 = [1] + [0 for i in range(1, self.requests_number + 1)], 0  # начальные условия
        self.st_names = [name for name in range(self.max_states)]
        self.ps = [[] for _ in range(self.max_states)]
        self.Y = np.array(0)
        self.tau = 0.01  # шаг интегрирования

    def f(self, p):
        n = self.n
        mu = self.mu
        lamda = self.lamda
        requests_number = self.requests_number

        rtrn = [-lamda * p[0] + n * mu * p[1]]
        rtrn += [(lamda * p[k - 1] - (lamda + n * mu) * p[k] + n * mu * p[k + 1])
                 for k in range(1, requests_number)]

        rtrn += [lamda * p[requests_number - 1] - (n * mu) * p[requests_number]]

        return rtrn

    def get_report(self):
        """ Построение графика вероятностей для состояний системы """

        for sys_state in range(len(self.ps)):
            plt.plot(self.ts, self.ps[sys_state], linewidth=1, label='state ' + str(self.st_names[sys_state]))

        print("Предельные значения распределения: ")
        for sys_state in range(self.max_states):
            print('state ' + str(sys_state) + ': ' + str(self.Y[-1][sys_state]))

        plt.title("График вероятностей состояний СМО")
        plt.grid()
        plt.legend()
        plt.show()

    def increment(self, y):
        k1 = self.mult(self.tau, self.f(y))
        k2 = self.mult(self.tau, self.f(self.add(y, self.mult(0.5 * self.tau, k1))))
        k3 = self.mult(self.tau, self.f(self.add(y, self.mult(0.5 * self.tau, k2))))
        k4 = self.mult(self.tau, self.f(self.add(y, self.mult(self.tau, k3))))

        rtrn = self.add(self.mult(1 / 6, k1), self.mult(1 / 3, k2))
        rtrn = self.add(rtrn, self.mult(1 / 3, k3))
        rtrn = self.add(rtrn, self.mult(1 / 6, k4))

        return rtrn

    def runge_kutta(self):
        self.ts.append(self.t0)
        self.ys.append(self.y0)

        cur_t = self.t0
        cur_y = self.y0

        for state in range(self.max_states):
            self.ps[state].append(cur_y[state])

        while cur_t < self.tmax:  # цикл по временному промежутку интегрирования
            self.tau = min(self.tau, self.tmax - cur_t)  # определение минимального шага self.tau
            cur_y = self.add(cur_y, self.increment(cur_y))  # расчёт значения в точке t0,y0 для задачи Коши
            cur_t = cur_t + self.tau  # приращение времени
            self.ts.append(cur_t)
            self.ys.append(cur_y)

            for state in range(self.max_states):
                self.ps[state].append(cur_y[state])

        self.Y = np.array(self.ys)

    def calc_lim_prob(self):

        print("Предельные значения распределения: ")
        for sys_state in range(self.max_states):
            print('state ' + str(sys_state) + ': ' + str(np.array(self.ys)[-1][sys_state]))

    def metrics(self):

        p = self.lamda / (self.n * self.mu)
        t_smo = 1 / (self.n * self.mu)
        p_rej = (p ** self.n * (1 - p)) / (1 - p ** (self.n + 1))
        Q = 1 - p_rej
        A = self.lamda * Q
        print('Среднее время прибывания заявки в СМО:', t_smo)
        print('Вероятность отказа:', p_rej)
        print('Отностиельная пропускная способность:', Q)
        print('Абсолютная пропускная способность:', A)

    @staticmethod
    def mult(element, array):
        # переопределение умножения для массивов
        for i in range(len(array)):
            array[i] *= element

        return array

    @staticmethod
    def add(array, array1):
        # переопределение сложения для массивов
        for i in range(len(array)):
            array[i] += array1[i]

        return array

    @staticmethod
    def arr_sum(array, n_from, n_to, degree):
        # сумма элементов массива
        total = 0
        for i in range(n_from, n_to + 1):
            if (i != 0) or (degree != 0):
                total += array[i] * (i ** degree)
            else:
                total += array[i]
        return total

    def plotting(self):
        plt.figure(figsize=[20, 10])
        for sys_state in range(len(self.ps)):
            plt.plot(self.ts, self.ps[sys_state], label='state ' + str(sys_state))

        plt.title("График вероятностей состояний СМО")
        plt.grid()
        plt.legend()
        plt.show()

    def run(self):
        print('Статистическая модель:')
        self.runge_kutta()
        self.calc_lim_prob()
        self.metrics()

        self.plotting()
