from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import numpy as np


class Complex(ABC):
    @staticmethod
    def polar(*args):
        return ComplexPolar(*args)

    @staticmethod
    def rect(*args):
        return ComplexRect(*args)

    @abstractmethod
    def to_rect(self) -> ComplexRect:
        pass

    @abstractmethod
    def to_polar(self) -> ComplexPolar:
        pass

    @abstractmethod
    def _apply(self, function) -> Complex:
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    def real(self):
        return self.to_rect().real

    @property
    def imag(self):
        return self.to_rect().imag

    def abs(self):
        return self.to_polar().abs()

    def angle(self):
        return self.to_polar().angle()

    def matmul(self, other) -> Complex:
        a = self.to_polar()
        b = other.to_polar()
        abs = a.abs().matmul(b.abs())
        ang = (a.angle() + b.angle().transpose(-1, -2)).sum(-1).unsqueeze(-1)
        return ComplexPolar(abs, ang)

    def mv(self, other) -> Complex:
        a = self.to_polar()
        b = other.to_polar()
        abs = a.abs().mv(b.abs())
        ang = (a.angle() + b.angle().T).sum(1)
        return ComplexPolar(abs, ang)

    def unsqueeze(self, dim) -> Complex:
        return self._apply(lambda x: x.unsqueeze(dim))

    def squeeze(self, dim=None) -> Complex:
        return self._apply(lambda x: x.squeeze(dim))

    def diag(self) -> Complex:
        diag = torch.diag_embed if len(self.shape) > 1 else torch.diag
        return self._apply(diag)

    def to(self, *args, **kwargs) -> Complex:
        return self._apply(lambda x: x.to(*args, **kwargs))

    @abstractmethod
    def conj(self) -> Complex:
        pass

    def __add__(self, other):
        a = self.to_rect()
        b = self.to_rect()
        return ComplexRect(a.real + b.real, a.imag + b.imag)

    def __sub__(self, other):
        a = self.to_rect()
        b = self.to_rect()
        return ComplexRect(a.real - b.real, a.imag - b.imag)


class ComplexPolar(Complex):
    def __init__(self, *args):
        self._abs: torch.Tensor
        self._angle: torch.Tensor
        if len(args) == 1:
            self._abs = args[0].abs()
            self._angle = args[0].angle()
        elif len(args) == 2:
            self._abs = args[0]
            self._angle = args[1]
        assert self._abs.shape == self._angle.shape

    def to_polar(self) -> ComplexPolar:
        return self

    def to_rect(self):
        return ComplexRect(self._abs * self._angle.cos(), self._abs * self._angle.sin())

    def _apply(self, function) -> Complex:
        return ComplexPolar(function(self._abs), function(self._angle))

    @property
    def shape(self):
        return self._abs.shape

    def abs(self) -> torch.Tensor:
        return self._abs

    def angle(self) -> torch.Tensor:
        return self._angle

    def conj(self) -> Complex:
        return ComplexPolar(self._abs, -self._angle)


class ComplexRect(Complex):
    def __init__(self, *args):
        self._real: torch.Tensor
        self._imag: torch.Tensor
        if len(args) == 1:
            self._real = args[0].real
            self._imag = args[0].imag
        elif len(args) == 2:
            self._real = args[0]
            self._imag = args[1]
        assert self.real.shape == self.imag.shape

    @staticmethod
    def from_numpy(array: np.ndarray) -> ComplexRect:
        return ComplexRect(torch.from_numpy(array.real), torch.from_numpy(array.imag))

    def to_polar(self):
        return ComplexPolar((self._real ** 2 + self._imag ** 2 + 1e-16).sqrt(),
                            (self._imag / (self._real + 1e-16)).atan())

    def to_rect(self):
        return self

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    @property
    def shape(self):
        return self.real.shape

    def _apply(self, function) -> Complex:
        return ComplexRect(function(self._real), function(self._imag))

    def conj(self) -> Complex:
        return ComplexPolar(self._real, -self._imag)
