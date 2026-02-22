import math

class Annealer:
    def __init__(self, total_steps, shape='linear', baseline=0.0, r_value = 0.5, cyclical=False, disable=False):

        self.current_step = 0

        if shape not in ['linear', 'cosine']:
            raise ValueError("Shape must be one of 'linear', 'cosine'")
        self.shape = shape

        if not 0 <= float(baseline) <= 1:
            raise ValueError("Baseline must be a float between 0 and 1.")
        self.baseline = baseline

        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("Argument total_steps must be an integer greater than 0")
        self.total_steps = total_steps

        self.cyclical = cyclical
        self.r_value = r_value
        self.disable = disable

    def __call__(self):
        return self._slope()

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def set_cyclical(self, value):
        if not isinstance(value, bool):
            raise ValueError("Argument to cyclical method must be a boolean.")
        self.cyclical = value
        return

    def _slope(self):
        if self.cyclical:
            progress = self.current_step / (self.total_steps*self.r_value)
        else:
            progress = self.current_step / (self.total_steps)
        progress = min(progress, 1.0)
        if self.shape == 'linear':
            y = (progress)
        elif self.shape == 'cosine':
            y = (1 - math.cos(math.pi * progress)) / 2
        else:
            y = 1.0
        y = self._add_baseline(y)
        return y

    def _add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out