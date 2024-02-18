import numpy as np
from module.parameters import Parameters

class Sigmoid:
    """Реализует сигмоиду"""

    def __init__(self):
        self.params = Parameters(0)  # У Sigmoid нет обучаемых параметров
        self.cache = None  # Для хранения входного тензора во время forward pass

    def forward(self, inpt):
        """Реализует forward-pass"""
        self.out = 1 / (1 + np.exp(-inpt))
        self.cache = inpt
        return self.out

    def __call__(self, *inpt):
        """Аналогично forward"""
        return self.forward(*inpt)

    def parameters(self):
        """Возвращает параметры модели"""
        return self.params

    def _zero_grad(self):
        """Обнуляет градиенты модели"""
        pass  # Не нужен для Sigmoid, так как нет обучаемых параметров

    def _compute_gradients(self, grads):
        """Считает градиенты модели"""
        sigmoid_grad = self.out * (1 - self.out)  # Производная сигмоиды
        input_grads = grads * sigmoid_grad
        return input_grads

    def _train(self):
        """Переводит модель в режим обучения"""
        pass  # Не требуется для Sigmoid

    def _eval(self):
        """Переводит модель в режим оценивания"""
        pass  # Не требуется для Sigmoid

    def __repr__(self):
        return "Sigmoid()"