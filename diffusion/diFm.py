import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def diffusion_model_with_data(initial_distribution, T, D):
    """
    Моделирует процесс диффузии с учетом реальных данных.

    Параметры:
        initial_distribution (numpy.ndarray): Начальное распределение концентрации.
        T (int): Количество временных шагов.
        D (float): Коэффициент диффузии.

    Возвращает:
        list: Список массивов, содержащих изменение концентрации во времени.
    """
    # Массив для хранения истории изменения концентрации
    history = [initial_distribution.copy()]

    # Вычисление изменения концентрации во времени
    for t in range(T):
        new_concentration = initial_distribution.copy()
        for i in range(1, initial_distribution.shape[0]-1):
            for j in range(1, initial_distribution.shape[1]-1):
                laplacian = (
                    initial_distribution[i+1, j] +
                    initial_distribution[i-1, j] +
                    initial_distribution[i, j+1] +
                    initial_distribution[i, j-1] -
                    4 * initial_distribution[i, j]
                )
                new_concentration[i, j] += D * laplacian
        initial_distribution = new_concentration
        history.append(initial_distribution.copy())

    return history

# Загрузка данных о распространении болезни
data = pd.read_csv("covid_data.csv")
cases = data["Количество зараженных"].values

# Использование данных для моделирования диффузии
initial_distribution = np.zeros((len(cases), 1))
initial_distribution[:, 0] = cases
T = 100  # Количество временных шагов
D = 0.1  # Коэффициент диффузии
history = diffusion_model_with_data(initial_distribution, T, D)

# Визуализация изменения концентрации во времени
plt.figure(figsize=(10, 5))
for i, concentration in enumerate(history):
    if i % 10 == 0:
        plt.plot(concentration, label=f"Time Step: {i}")
plt.xlabel("Дни")
plt.ylabel("Количество зараженных")
plt.title("Распространение COVID-19 во времени")
plt.legend()
plt.show()
