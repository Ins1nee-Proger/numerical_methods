import csv
import numpy as np
import matplotlib.pyplot as plt

# 1. Зчитування даних
def read_data(filename):
    x, y = [], []

    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)

        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))

    return np.array(x), np.array(y)


# 2. Розділені різниці Ньютона
def divided_differences(x, y):
    n = len(x)
    coef = np.copy(y)

    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])

    return coef


# 3. Поліном Ньютона
def newton_polynomial(coef, x_data, x):
    result = coef[-1]

    for k in range(len(coef)-2, -1, -1):
        result = result * (x - x_data[k]) + coef[k]

    return result


def newton_predict(x, y, xp):
    coef = divided_differences(x, y)
    return newton_polynomial(coef, x, xp)


def lagrange(x, y, xp):
    n = len(x)
    yp = 0

    for i in range(n):
        L = 1
        for j in range(n):
            if i != j:
                L *= (xp - x[j]) / (x[i] - x[j])
        yp += y[i] * L

    return yp


# MAIN
x_data, y_data = read_data("data.csv")

dd = divided_differences(x_data, y_data)

print("\nКоефіцієнти розділених різниць Ньютона:")
for i, v in enumerate(dd):
    print(f"f[x0..x{i}] = {v:.6f}")
predict_point = 6000

print("\n=== ПРОГНОЗ ===")
print("Ньютон:", newton_predict(x_data, y_data, predict_point))
print("Лагранж:", lagrange(x_data, y_data, predict_point))


# 5. ГРАФІК ІНТЕРПОЛЯЦІЇ
x_plot = np.linspace(min(x_data), max(x_data), 400)

coef = divided_differences(x_data, y_data)
y_plot = [newton_polynomial(coef, x_data, x) for x in x_plot]

plt.figure()
plt.scatter(x_data, y_data, label="Експериментальні точки")
plt.plot(x_plot, y_plot, label="Поліном Ньютона")
plt.scatter(
    predict_point,
    newton_predict(x_data, y_data, predict_point),
    marker='x',
    s=120,
    label="Прогноз"
)

plt.title("Інтерполяція Ньютона")
plt.xlabel("n")
plt.ylabel("t (мс)")
plt.legend()
plt.grid()
plt.show()


# 6. ДОСЛІДЖЕННЯ ВПЛИВУ КРОКУ
print("\n=== ВПЛИВ КРОКУ ===")

for nodes in [3, 4, 5]:
    x_sub = x_data[:nodes]
    y_sub = y_data[:nodes]

    pred = newton_predict(x_sub, y_sub, predict_point)
    print(f"{nodes} вузлів -> прогноз:", pred)


# 7. ВПЛИВ ІНТЕРВАЛУ
print("\n=== ВПЛИВ ІНТЕРВАЛУ ===")

step = 2000

for end in [6000, 10000, 16000]:
    x_sub = np.arange(1000, end + 1, step)
    y_sub = np.interp(x_sub, x_data, y_data)

    pred = newton_predict(x_sub, y_sub, predict_point)
    print(f"Інтервал до {end} -> прогноз:", pred)


# 8. АНАЛІЗ ЕФЕКТУ РУНГЕ
plt.figure()
print("\n=== АНАЛІЗ ПОХИБКИ ===")

true_value = newton_predict(x_data, y_data, predict_point)

for nodes in [3, 4, 5]:
    x_sub = x_data[:nodes]
    y_sub = y_data[:nodes]

    pred = newton_predict(x_sub, y_sub, predict_point)
    error = abs(true_value - pred)

    print(f"{nodes} вузлів -> похибка:", error)

for nodes in [3, 4, 5]:
    x_sub = x_data[:nodes]
    y_sub = y_data[:nodes]

    coef = divided_differences(x_sub, y_sub)
    y_plot = [newton_polynomial(coef, x_sub, x) for x in x_plot]

    plt.plot(x_plot, y_plot, label=f"{nodes} вузлів")

plt.scatter(x_data, y_data)
plt.title("Ефект Рунге")
plt.legend()
plt.grid()
plt.show()


# 9. ГРАФІК ПОХИБКИ
plt.figure()

nodes_list = list(range(3, min(21, len(x_data) + 1)))
errors = []

true_value = newton_predict(x_data, y_data, predict_point)

for nodes in nodes_list:
    x_sub = x_data[:nodes]
    y_sub = y_data[:nodes]
    pred = newton_predict(x_sub, y_sub, predict_point)
    errors.append(abs(true_value - pred))

plt.plot(nodes_list, errors, marker='o')
plt.title("Графік похибки інтерполяції")
plt.xlabel("Кількість вузлів")
plt.ylabel("Абсолютна похибка")
plt.grid()
plt.show()


# 10. ГРАФІКИ ДЛЯ 10, 15 ТА 20 ВУЗЛІВ
for nodes in [10, 15, 20]:
    if nodes <= len(x_data):
        idx = np.linspace(0, len(x_data) - 1, nodes, dtype=int)
        x_sub = x_data[idx]
        y_sub = y_data[idx]

        coef = divided_differences(x_sub, y_sub)
        y_plot = [newton_polynomial(coef, x_sub, x) for x in x_plot]

        plt.figure()
        plt.scatter(x_data, y_data, label="Експериментальні точки")
        plt.plot(x_plot, y_plot, label=f"Поліном Ньютона ({nodes} вузлів)")
        plt.scatter(
            predict_point,
            newton_predict(x_sub, y_sub, predict_point),
            marker='x',
            s=120,
            label="Прогноз"
        )

        plt.title(f"Інтерполяція Ньютона ({nodes} вузлів)")
        plt.xlabel("n")
        plt.ylabel("t (мс)")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print(f"Недостатньо точок для {nodes} вузлів")