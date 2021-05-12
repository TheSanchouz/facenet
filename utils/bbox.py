class BoundingBox:
    def __init__(
            self,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            score: float,
    ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score


class AgeGenderBoundingBox:
    def __init__(
            self,
            box: BoundingBox,
            age: int,
            gender: dict
    ):
        self.box = box
        self.age = age
        self.gender = gender
