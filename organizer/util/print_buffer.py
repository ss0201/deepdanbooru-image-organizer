class PrintBuffer:
    buffer = ""

    def add(self, obj: object) -> None:
        self.buffer += f"{obj}\n"

    def print(self) -> None:
        print(self.buffer, flush=True)
        self.buffer = ""
