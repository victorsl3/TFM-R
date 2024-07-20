import math
from hyperbolic.poincare import Point

class PoincareBall:
    def __init__(self, radius=1):
        self.radius = radius  # El radio típico para el disco de Poincaré es 1.

    @staticmethod
    def dot(x, y):
        return x.x * y.x + x.y * y.y

    @staticmethod
    def norm(v):
        return math.sqrt(v.x**2 + v.y**2)

    @staticmethod
    def subtract(p1, p2):
        return Point(p1.x - p2.x, p1.y - p2.y)

    @staticmethod
    def exp_map(base_point, tangent_vector):
        norm_v = PoincareBall.norm(tangent_vector)
        if norm_v == 0:
            return base_point
        return Point(
            math.tanh(0.5 * norm_v) / norm_v * tangent_vector.x,
            math.tanh(0.5 * norm_v) / norm_v * tangent_vector.y
        )

    @staticmethod
    def log_map(base_point, other_point):
        diff = PoincareBall.subtract(other_point, base_point)
        norm_diff = PoincareBall.norm(diff)
        if norm_diff == 0:
            return Point(0, 0)
        return Point(
            2 * math.atanh(norm_diff) / norm_diff * diff.x,
            2 * math.atanh(norm_diff) / norm_diff * diff.y
        )
    
    def mobius_addition(self, x, y):
        numerator = self.add(self.multiply(x, 1 - self.dot(y, y)), y)
        denominator = 1 - 2 * self.dot(x, y) + self.dot(x, x) * self.dot(y, y)
        return self.divide(numerator, denominator)
    
    @staticmethod
    def add(p1, p2):
        return Point(p1.x + p2.x, p1.y + p2.y)
    
    @staticmethod
    def multiply(p, scalar):
        return Point(p.x * scalar, p.y * scalar)
    
    @staticmethod
    def divide(p, scalar):
        if scalar != 0:
            return Point(p.x / scalar, p.y / scalar)
        else:
            raise ValueError("División por cero no está permitida.")
    
    def geodesic_distance(self, x, y):
        diff = self.subtract(x, y)
        diff_norm = self.norm(diff)
        return 2 * math.atanh(diff_norm)
