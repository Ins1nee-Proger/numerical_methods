import numpy as np
import matplotlib.pyplot as plt
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)
def dM_analytical(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# Точка обчислення
t0 = 1.0
exact_val = dM_analytical(t0)

# 2. Формула центральної різниці для апроксимації
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Дослідження залежності похибки від кроку h (пункт 2)
print(f"Точне значення похідної в t0={t0}: {exact_val:.10f}\n")
print("h\t\tАпроксимація\tПохибка")
for p in range(1, 11):
    h_test = 10**(-p)
    approx = central_diff(M, t0, h_test)
    error = abs(approx - exact_val)
    print(f"1e-{p}\t{approx:.10f}\t{error:.10f}")

# 3. Приймаємо крок h = 10^-3 (згідно з пунктом 3)
h = 1e-3

# 4. Обчислення для двох кроків: h та 2h
yh = central_diff(M, t0, h)
y2h = central_diff(M, t0, 2*h)

# 5. Похибка при кроці h
R1 = abs(yh - exact_val)

# 6. Метод Рунге-Ромберга (уточнення за двома кроками)
# Для центральної різниці порядок p=2, тому дільник 2^2 - 1 = 3
y_runge = yh + (yh - y2h) / 3
R2 = abs(y_runge - exact_val)

# 7. Метод Ейткена (уточнення за трьома кроками: h, 2h, 4h)
y4h = central_diff(M, t0, 4*h)

# Уточнене значення за Ейткеном
numerator = (y2h)**2 - y4h * yh
denominator = 2*y2h - (y4h + yh)
y_aitken = numerator / denominator

# Оцінка порядку точності p
p_est = np.log(abs((y4h - y2h) / (y2h - yh))) / np.log(2)
R3 = abs(y_aitken - exact_val)

# Виведення результатів
print("-" * 50)
print(f"Результати для h = {h}:")
print(f"y'(h):         {yh:.10f}, Похибка R1:", R1)
print(f"Метод Рунге:   {y_runge:.10f}, Похибка R2: {R2:.10f}")
print(f"Метод Ейткена: {y_aitken:.10f}, Похибка R3: {R3:.10f}")
print(f"Оцінений порядок точності p: {p_est:.4f}")
# 1. Графік залежності похибки від кроку h (Пункт 2)


# Підготовка спільних даних
t_plot = np.linspace(0, 20, 1000)
M_plot = M(t_plot)
dM_plot = dM_analytical(t_plot)

#ФІЗИЧНА МОДЕЛЬ (ФУНКЦІЯ ТА ПОХІДНА)

plt.figure(figsize=(12, 5))
#Графік вологості
plt.subplot(1, 2, 1)
plt.plot(t_plot, M_plot, 'b-', label='M(t)')
plt.scatter([t0], [M(t0)], color='red', zorder=5, label=f't0={t0}')
plt.title('Вологість ґрунту M(t)')
plt.xlabel('Час t')
plt.ylabel('%')
plt.grid(True, alpha=0.3)
plt.legend()

#Графік аналітичної похідної
plt.subplot(1, 2, 2)
plt.plot(t_plot, dM_plot, 'g-', label="M'(t)")
plt.axhline(0, color='black', lw=1, ls='--')
plt.scatter([t0], [exact_val], color='red', zorder=5)
plt.title('Швидкість зміни вологості')
plt.xlabel('Час t')
plt.ylabel("M'(t)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

#АНАЛІЗ ТОЧНОСТІ (КРОК ТА МЕТОДИ)
plt.figure(figsize=(12, 5))
# 3. Дослідження кроку h
h_test = [10**(-p) for p in range(1, 16)]
err_test = [abs(central_diff(M, t0, ht) - exact_val) for ht in h_test]

plt.subplot(1, 2, 1)
plt.loglog(h_test, err_test, 'o-', color='purple', markersize=4)
plt.axvline(x=1e-5, color='orange', ls='--', label='h_opt ≈ 10⁻⁵')
plt.title('Залежність похибки від кроку h')
plt.xlabel('Крок h (log)')
plt.ylabel('Похибка (log)')
plt.grid(True, which="both", alpha=0.2)
plt.legend()

# Порівняння методів
methods_list = ['Центральна', 'Рунге-Ромберг', 'Ейткен']
errors_list = [R1, R2, R3]

plt.subplot(1, 2, 2)
bars = plt.bar(methods_list, errors_list, color=['#3498db', '#2ecc71', '#f1c40f'], alpha=0.7)
plt.yscale('log')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1e}', va='bottom', ha='center')
plt.title(f'Похибка методів (h = {h})')
plt.ylabel('Абсолютна похибка (log)')
plt.tight_layout()
plt.show()