class PrintBuffer:
    buffer = ""

    def add(self, string: str) -> None:
        self.buffer += f"{string}\n"

    def print(self) -> None:
        print(self.buffer, flush=True)
        self.buffer = ""
