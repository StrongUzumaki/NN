import time

def progress_bar(iterable, text='Epoch progress', end=''):
    """Мониториг выполнения эпохи

    ---------
    Параметры
    ---------
    iterable
        Что-то по чему можно итерироваться

    text: str (default='Epoch progress')
        Текст, выводящийся в начале

    end : str (default='')
        Что вывести в конце выполнения
    """
    max_num = len(iterable)
    iterable = iter(iterable)

    start_time = time.time()
    cur_time = 0
    approx_time = 0

    print('\r', end='')

    it = 0
    while it < max_num:
        it += 1
        print(f"{text}: [", end='')

        progress = int((it - 1) / max_num * 50)
        print('=' * progress, end='')
        if progress != 50:
            print('>', end='')
            print(' ' * (50 - progress - 1), end='')
        print('] ', end='')

        print(f'{it - 1}/{max_num}', end='')
        print(' ', end='')

        print(f'{cur_time}s>{approx_time}s', end='')

        yield next(iterable)

        print('\r', end='')
        print(' ' * (60 + len(text) + len(str(max_num)) + len(str(it)) \
                     + len(str(cur_time)) + len(str(approx_time))),
              end='')
        print('\r', end='')

        cur_time = time.time() - start_time

        approx_time = int(cur_time / it * (max_num - it))
        cur_time = int(cur_time)
        print(end, end='')

def gradient_check(x, y, neural_net, loss_function, epsilon=1e-7):
    """
    Выполняет проверку градиента для всех параметров нейронной сети.
    
    Параметры:
    x : np.ndarray
        Входные данные сети.
    y : np.ndarray
        Целевые значения для входных данных.
    neural_net : объект нейронной сети
        Нейронная сеть с методами forward и backward.
    loss_function : функция
        Функция потерь, используемая для обучения сети.
    epsilon : float, optional
        Малое значение для вычисления численного градиента.
    
    Возвращает:
    bool
        True, если проверка градиента успешна, иначе False.
    """
    # Инициализация
    param_grads = neural_net.get_params_grads()  # Получаем градиенты параметров
    numerical_grads = np.zeros_like(param_grads)
    
    # Вычисление численного градиента для каждого параметра
    for i in range(len(param_grads)):
        old_value = param_grads[i]
        
        # Увеличение параметра на epsilon и вычисление потерь
        param_grads[i] += epsilon
        loss_plus_epsilon = loss_function(neural_net.forward(x), y)
        
        # Уменьшение параметра на 2*epsilon и вычисление потерь
        param_grads[i] = old_value - epsilon
        loss_minus_epsilon = loss_function(neural_net.forward(x), y)
        
        # Восстановление исходного значения параметра
        param_grads[i] = old_value
        
        # Вычисление численного градиента
        numerical_grads[i] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
    
    # Сравнение численных и аналитических градиентов
    relative_error = np.linalg.norm(param_grads - numerical_grads) / (np.linalg.norm(param_grads) + np.linalg.norm(numerical_grads))
    
    print("Relative error:", relative_error)
    
    return relative_error < 1e-7