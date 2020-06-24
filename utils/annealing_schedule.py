
class AnnealingSchedule:
    # linear annealing of a parameter
    def __init__(self, starting_par: float, ending_par: float, n_steps: int, inverse=False):
        self.starting_p = starting_par
        self.ending_p = ending_par
        self.n_steps = n_steps
        self.p_drop = (self.starting_p - self.ending_p) / self.n_steps
        self.current_p = starting_par

        self.inverse = inverse

    def anneal(self):
        if not self.inverse:
            if self.current_p > self.ending_p:
                self.current_p = max(self.ending_p, self.current_p - self.p_drop)
        else:

            if self.current_p < self.ending_p:
                self.current_p = self.current_p - self.p_drop

        return self.current_p
