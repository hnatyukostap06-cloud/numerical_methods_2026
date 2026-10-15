import numpy as np
import math

# 1. Аналітичне розв'язання
# Задана функція вологості ґрунту
def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

# Її аналітична похідна
def M_prime_exact(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

t0 = 1.0
exact_val = M_prime_exact(t0)
print(f"1. Точне значення похідної y'(x0) при t0=1: {exact_val}")

# 2. Дослідження залежності похибки від кроку h
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Дослідження  діапазону h 
h_values = np.logspace(-20, 3, 1000)
best_h = None
min_error = float('inf')

for h in h_values:
    try:
        approx_val = central_diff(M, t0, h)
        error = abs(approx_val - exact_val)
        if error < min_error:
            min_error = error
            best_h = h
    except ZeroDivisionError:
        pass

print(f"2. Оптимальний крок h0: {best_h:.1e}")
print(f"   Досягнута точність R0: {min_error:.2e}")

# 3. Приймаємо фіксований крок
h_fixed = 1e-3
print(f"3. Прийнято значення кроку h = {h_fixed}")

# 4. Обчислення з двома кроками (h та 2h)
y_prime_h = central_diff(M, t0, h_fixed)
y_prime_2h = central_diff(M, t0, 2 * h_fixed)
print(f"4. Похідна з кроком h:  {y_prime_h}")
print(f"   Похідна з кроком 2h: {y_prime_2h}")

# 5. Обчислення похибки при кроці h
R1 = abs(y_prime_h - exact_val)
print(f"5. Похибка R1: {R1}")

# 6. Метод Рунге-Ромберга
y_R = y_prime_h + (y_prime_h - y_prime_2h) / 3
R2 = abs(y_R - exact_val)
print(f"6. Метод Рунге-Ромберга, уточнене значення: {y_R}")
print(f"   Похибка R2: {R2}")
print(f"   Характер зміни похибки: похибка зменшилась порівняно з R1.")

# 7. Метод Ейткена
y_prime_4h = central_diff(M, t0, 4 * h_fixed)

y_E = (y_prime_2h**2 - y_prime_4h * y_prime_h) / (2 * y_prime_2h - (y_prime_4h + y_prime_h))
p = (1 / math.log(2)) * math.log(abs((y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h)))
R3 = abs(y_E - exact_val)

print(f"7. Метод Ейткена, уточнене значення: {y_E}")
print(f"   Порядок точності формули p: {p}")
print(f"   Похибка R3: {R3}")
print(f"   Характер зміни похибки: похибка R3 суттєво менша за R1 і приблизно відповідає рівню точності методу Рунге-Ромберга.")