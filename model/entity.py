from math import sqrt


class Vec3(object):
    x: float
    y: float
    z: float

    def __init__(self, xx, yy, zz):
        self.x = xx
        self.y = yy
        self.z = zz

    def scalar_mul(self, other):
        return Vec3(self.x*other, self.y*other, self.z*other)

    def as_tuple(self):
        return self.x, self.y, self.z

    @staticmethod
    def length(vec):
        return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z)

    @staticmethod
    def normalize(vec):
        vec_magnitude = Vec3.length(vec)
        if vec_magnitude == 0:
            return vec
        return Vec3(vec.x/vec_magnitude, vec.y/vec_magnitude, vec.z/vec_magnitude)

    @staticmethod
    def dot(a, b):
        return a.x * b.x + a.y * b.y + a.z*b.z

    def pair_wise_op(self, other, op):
        return Vec3(op(self.x, other.x), op(self.y, other.y), op(self.z, other.z))

    def constant_op(self, op):
        return Vec3(op(self.x), op(self.y), op(self.z))

    def __add__(self, other):
        if type(other) == Vec3:
            return self.pair_wise_op(other, lambda x, y: float(x + y))
        elif type(other) == float or type(other) == int:
            return self.constant_op(lambda x: float(x+other))

    def __sub__(self, other):
        if type(other) == Vec3:
            return self.pair_wise_op(other, lambda x, y: float(x - y))
        elif type(other) == float or type(other) == int:
            return self.constant_op(lambda x: float(x - other))

    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            return self.constant_op(lambda x: float(x * other))

    def __str__(self):
        return "Vec3({}, {}, {})".format(self.x, self.y, self.z)

    @staticmethod
    def clamp(vec, max_value):
        mag = Vec3.length(vec)
        if mag <= max_value:
            return vec
        return Vec3.normalize(vec) * max_value

    @staticmethod
    def clamp_d(vec, min_value):
        mag = Vec3.length(vec)
        if mag >= min_value:
            return vec
        return Vec3.normalize(vec) * min_value

    @staticmethod
    def clamp_d_u(vec, min_value, max_value):
        return Vec3.clamp(Vec3.clamp_d(vec, min_value), max_value)


class Vec2(object):
    x: float
    y: float

    def __init__(self, xx, yy):
        self.x = xx
        self.y = yy

    def length(self):
        return sqrt(self.x*self.x + self.y*self.y)

    def __add__(self, other):
        if type(other) == Vec2:
            return Vec2(self.x + other.x, self.y + other.y)
        else:
            return Vec2(self.x + other, self.y + other)

    def __sub__(self, other):
        if type(other) == Vec2:
            return Vec2(self.x - other.x, self.y - other.y)
        else:
            return Vec2(self.x - other, self.y - other)

    def dot(self, other):
        return self.x * other.x + self.y + other.y

    def __mul__(self, other):
        return Vec2(other * self.x, self.y * other)

    def normalize(self):
        mag = self.length()
        return Vec2(self.x/mag, self.y/mag)


class Entity(object):
    position: Vec3
    velocity: Vec3
    radius: float
    radius_change_speed: float
    arena_e: float
    radius: float
    mass: float


