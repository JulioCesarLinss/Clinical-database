import time

# Conta quantos elementos de uma lista satisfazem uma condição
def count(testFn, items):
    s = 0
    for x in items:
        if testFn(x):
            s += 1
    return s


# Mede o tempo de execução de cada etapa do pipeline do FDC
class Timing:
    def __init__(self, name="Duration"):
        self.name = name
        self.tStart = time.process_time()        # tempo total desde o início
        self.tStepStart = time.process_time()    # tempo desde o início da etapa atual

    # Registra o tempo da etapa atual e imprime no console
    def step(self, message=""):
        now = time.process_time()
        duration = now - self.tStart             # tempo total acumulado
        durationStep = now - self.tStepStart     # tempo só dessa etapa
        self.tStepStart = now                    # reinicia o contador da próxima etapa

        if message == "":
            print(f"{self.name}: {durationStep:0.5f} / {duration:0.3f}s")
        else:
            print(f"{self.name} ({message}): {durationStep:0.5f} / {duration:0.3f}s")
        return duration