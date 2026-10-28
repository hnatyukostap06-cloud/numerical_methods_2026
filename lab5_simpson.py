import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, erf, log

# Функція навантаження сервера

def f(x):
    x = np.asarray(x)
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a, b = 0.0, 24.0
EPS_TARGET = 1e-8
ADAPTIVE_EPS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

# Аналітичне значення інтеграла
I0 = 1200 + 5 * sqrt(pi / 0.2) * erf(sqrt(0.2) * 12)


def simpson(N):
    """Складова формула Сімпсона. Кількість розбиттів N має бути парною."""
    if N < 2 or N % 2 != 0:
        raise ValueError("N має бути парним числом, не меншим за 2.")
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return h / 3 * (
        y[0] + y[-1]
        + 4 * np.sum(y[1:-1:2])
        + 2 * np.sum(y[2:-1:2])
    )


def adaptive_simpson(eps, max_depth=50):
    """Адаптивний метод Сімпсона. Повертає інтеграл і кількість обчислень f(x)."""
    count = 0

    def ff(x):
        nonlocal count
        count += 1
        return float(f(x))

    def S(x0, x1, y0, ym, y1):
        return (x1 - x0) / 6 * (y0 + 4 * ym + y1)

    def recurse(x0, x1, y0, ym, y1, old, tol, depth):
        xm = (x0 + x1) / 2
        xl, xr = (x0 + xm) / 2, (xm + x1) / 2
        yl, yr = ff(xl), ff(xr)
        left = S(x0, xm, y0, yl, ym)
        right = S(xm, x1, ym, yr, y1)
        delta = left + right - old

        if depth <= 0 or abs(delta) <= 15 * tol:
            return left + right + delta / 15

        return (
            recurse(x0, xm, y0, yl, ym, left, tol / 2, depth - 1)
            + recurse(xm, x1, ym, yr, y1, right, tol / 2, depth - 1)
        )

    ya, yb = ff(a), ff(b)
    mid = (a + b) / 2
    ym = ff(mid)
    whole = S(a, b, ya, ym, yb)
    result = recurse(a, b, ya, ym, yb, whole, eps, max_depth)
    return result, count


def print_result(label, value, exact=I0):
    print(f"{label:<34} Значення: {value:.12f} | Абсолютна похибка: {abs(value - exact):.6e}")


def main():
    # Завдання 1
    print("\nЗАВДАННЯ 1. Задана функція та інтервал")
    print("Функція: f(x) = 50 + 20·sin(πx/12) + 5·exp(-0.2·(x−12)²)")
    print(f"Межі інтегрування: a = {a:g}, b = {b:g}")

    # Завдання 2
    print("\nЗАВДАННЯ 2. Точне значення інтеграла")
    print(f"Еталонне значення I0 = {I0:.12f}")

    # Завдання 3
    print("\nЗАВДАННЯ 3. Складова формула Сімпсона")
    print(f"Контрольне наближення при N = 24: I(24) = {simpson(24):.12f}")
    print(f"Абсолютна похибка при N = 24: {abs(simpson(24) - I0):.6e}")

    # Завдання 4
    print("\nЗАВДАННЯ 4. Дослідження похибки та пошук Nopt")
    ns = np.arange(10, 1001, 2)
    errors = np.array([abs(simpson(int(n)) - I0) for n in ns])
    acceptable = ns[errors <= EPS_TARGET]
    Nopt = int(acceptable[0]) if len(acceptable) else None
    print(f"Задана точність: ε = {EPS_TARGET:.0e}")
    if Nopt is not None:
        Iopt = simpson(Nopt)
        print(f"Оптимальна кількість розбиттів Nopt = {Nopt}")
        print(f"Наближене значення I(Nopt) = {Iopt:.12f}")
        print(f"Фактична абсолютна похибка = {abs(Iopt - I0):.6e}")
    else:
        print("У дослідженому діапазоні не знайдено N, що задовольняє задану точність.")

    plt.figure(figsize=(9, 5.5))
    plt.semilogy(ns, np.maximum(errors, np.finfo(float).tiny), label="Абсолютна похибка")
    plt.axhline(EPS_TARGET, linestyle="--", label=f"Задана точність {EPS_TARGET:.0e}")
    if Nopt is not None:
        plt.axvline(Nopt, linestyle=":", label=f"Nopt = {Nopt}")
    plt.xlabel("Кількість розбиттів N")
    plt.ylabel("Абсолютна похибка")
    plt.title("Залежність похибки формули Сімпсона від N")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("simpson_error.png", dpi=160)
    plt.close()
    print("Графік збережено у файлі simpson_error.png")

    # Завдання 5. N0 має бути кратним 4, щоб можна було використати N0/2 та N0/4.
    N0 = 24
    I_N0 = simpson(N0)
    eps0 = abs(I_N0 - I0)
    h0 = (b - a) / N0
    print("\nЗАВДАННЯ 5. Початкове наближення та похибка")
    print(f"Кількість розбиттів N0 = {N0}")
    print(f"Крок розбиття h = {h0:.6f}")
    print(f"Наближене значення I(N0) = {I_N0:.12f}")
    print(f"Абсолютна похибка = {eps0:.6e}")

    # Завдання 6. Метод Рунге—Ромберга для формули Сімпсона (порядок 4).
    I_half = simpson(N0 // 2)
    I_R = I_N0 + (I_N0 - I_half) / 15
    print("\nЗАВДАННЯ 6. Уточнення методом Рунге—Ромберга")
    print(f"I(N0) = {I_N0:.12f}")
    print(f"I(N0/2) = {I_half:.12f}")
    print_result("Уточнене значення I_R:", I_R)

    # Завдання 7. Метод Ейткена за формулами методички.
    I_quarter = simpson(N0 // 4)
    denominator = 2 * I_half - I_N0 - I_quarter
    print("\nЗАВДАННЯ 7. Уточнення методом Ейткена")
    print(f"I(N0) = {I_N0:.12f}")
    print(f"I(N0/2) = {I_half:.12f}")
    print(f"I(N0/4) = {I_quarter:.12f}")
    if abs(denominator) < 1e-14 or abs(I_half - I_N0) < 1e-14:
        print("Не вдалося надійно визначити уточнення або порядок методу для цих наближень.")
        I_E = float("nan")
        p = float("nan")
    else:
        I_E = (I_half**2 - I_N0 * I_quarter) / denominator
        ratio = abs((I_quarter - I_half) / (I_half - I_N0))
        p = log(ratio, 2) if ratio > 0 else float("nan")
        print_result("Уточнене значення I_E:", I_E)
        print(f"Оцінений порядок методу p = {p:.6f}")

    # Завдання 8. Порівняння всіх основних методів.
    print("\nЗАВДАННЯ 8. Порівняння результатів методів")
    methods = [
        (f"Сімпсон, N={N0}", I_N0),
        ("Рунге—Ромберга", I_R),
    ]
    if np.isfinite(I_E):
        methods.append(("Ейткен", I_E))
    print(f"{'Метод':<24} {'Наближене значення':>22} {'Абсолютна похибка':>22}")
    print("-" * 70)
    for name, value in methods:
        print(f"{name:<24} {value:>22.12f} {abs(value - I0):>22.6e}")
    print("Примітка: похибка обчислена порівнянням з еталонним значенням інтеграла.")

    # Завдання 9. Адаптивний алгоритм.
    print("\nЗАВДАННЯ 9. Адаптивний метод Сімпсона")
    print(f"{'Задана ε':<14} {'Наближене значення':>22} {'Фактична похибка':>20} {'Обчислень f(x)':>18}")
    print("-" * 78)
    for eps in ADAPTIVE_EPS:
        result, count = adaptive_simpson(eps)
        print(f"{eps:<14.0e} {result:>22.12f} {abs(result - I0):>20.6e} {count:>18}")

if __name__ == "__main__":
    main()
