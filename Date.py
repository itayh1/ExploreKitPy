
from datetime import datetime

class Date:

    def __init__(self):
        self.time = datetime.now()

    def getSeconds(self) -> int:
        return self.time.second.real

    def __sub__(self, other):
        return self.time - other.time