
class ProgressBar():
    def __init__(self):
        self.length = 30
        self.fill = '█'
        self.empty = '-'
        print(f'Progress: {self.empty * self.length} | 0/0 | {0:.1f}%', end='\r')

    def update(self, filled, total):
        percentage = min([filled, total]) / total
        filled_length = int(percentage * self.length)
        empty_length = self.length - filled_length
        print(f'Progress: {self.fill * filled_length}{self.empty * empty_length} | {filled}/{total} | {(percentage * 100):.1f}%', end='\r')
        if percentage >= 1:
            print()