class Monitor:
    def send_input(self, input):
        raise NotImplementedError

    def read_response():
        raise NotImplementedError


class PexpectMonitor(Monitor):
    def __init__(self, proc):
        self.proc = proc
    
    def send_input(self, input_marker, input):
        self.proc.expect(input_marker)
        self.proc.send(input + '\n')

    def read_response(self):
        self.proc.expect("Result: ")
        response = self.proc.readline().decode().strip()
        return response

    def finish(self):
        self.proc.close()
