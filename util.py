import time


class ProgressBar():
    def __init__(self, title: str='Progress'):
        self.title = title
        self.length = 30
        self.fill = '█'
        self.empty = '-'
        self.start = time.time()
        print(f'{self.title}: {self.empty * self.length} | 0/0 | {0:.1f}%', end='\r')

    def update(self, filled, total):
        percentage = min([filled, total]) / total
        filled_length = int(percentage * self.length)
        empty_length = self.length - filled_length
        elapsed = time.time() - self.start
        final = elapsed / percentage
        remaining = final - elapsed
        print(
            f'{self.title}: {self.fill * filled_length}{self.empty * empty_length} | {filled}/{total} | {(percentage * 100):.1f}% | '
            f'{(elapsed // 60):02.0f}:{(elapsed % 60):02.0f}, {(remaining // 60):02.0f}:{(remaining % 60):02.0f} remaining', end='\r')
        if percentage >= 1:
            print()


def pretty_print(object):
    def _format(object, level: int=0):
        if isinstance(object, dict):
            formatted = ''
            for key, value in object.items():
                formatted += f'\n{chr(9) * level}{key}: {_format(value, level+1)}'
            return formatted
        else:
            return f'{object}'
    print(_format(object)[1:])  # Delete first \n character