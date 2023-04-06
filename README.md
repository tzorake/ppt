# PPT1

## Как собрать и запустить

У меня установлены `Windows 10` и `g++ (MinGW.org GCC-6.3.0-1) 6.3.0`. \
Ниже описаны шаги по установке и запуску.

```batch
g++ main.cpp -fopenmp -o main.exe
main.exe
```

## Отчет

### Задание 1
> Реализуйте последовательный алгоритм решения квадратной системы линейных уравнений матричным методом. Для заполнения матрицы и вектора элементами можно воспользоваться генератором случайных чисел. Элементы матрицы и вектора – натуральные числа: a_ij, b_k ∈ (0, ... ,10]; i, j, k = 1, ... ,N. Количество уравнений N на усмотрение студента, в зависимости от сложности вычисления.

Реализован однопоточный метод решения СЛАУ, результат работы кода с консоли: \
![single](https://user-images.githubusercontent.com/9623983/230192920-00a530d8-8b9f-4d1a-9c95-e52e18b42c67.png)

### Задание 2
> Реализуйте выполнение операций согласно заданию 1, статическим методом многопоточной обработки. Число потоков задается параметром M.

Реализован статический метод многопоточной обработки решения СЛАУ, результат работы кода с консоли: \
![multi](https://user-images.githubusercontent.com/9623983/230192936-bf669e7c-c7b0-4bb7-95d4-b438a62f9bc9.png)

### Задание 3
> Выполните анализ эффективности и ускорения многопоточной обработки при разных параметрах N и M. Результаты представьте в табличной форме и графической форме.

Ранее составленные алгоритмы протестированы указанным в задание методом, результат работы кода с консоли: \
![test_algs](https://user-images.githubusercontent.com/9623983/230194685-21e6dcc0-c1b0-4b34-b30a-420b424ca9c1.png)

### Задание 5
> Исследуйте эффективность параллелизма при динамической декомпозиции. Сравните с эффективностью статической декомпозиции.

Ранее составленные алгоритмы протестированы для статической и динамической декомпозиции (слева и права соответственно), результат работы кода с консоли: \
![test_algs_sch](https://user-images.githubusercontent.com/9623983/230198807-a7dffc0d-ac98-4558-8a37-34149e84c497.png)