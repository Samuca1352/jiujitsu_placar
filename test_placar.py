from types import SimpleNamespace as P

import cv2
import numpy as np

from placar import CORES, PULAR, Placar, contar_dedos, escolher, frac_faixa, quem

# Placar: segura 0.5s pontua uma vez; só pontua de novo depois de abaixar
p = Placar(segurar=0.5, soltar=0.3)
assert [p.update(("cor", 2), t / 10) for t in range(10)].count(("cor", 2)) == 1
assert p.pontos == {"cor": 2, "outro": 0}
for t in range(10, 15):
    p.update(None, t / 10)
for t in range(15, 25):
    p.update(("outro", 3), t / 10)
assert p.pontos == {"cor": 2, "outro": 3}
p.update(("cor", 4), 2.5)  # gesto piscando rápido não pontua
p.update(None, 2.6)
assert p.pontos == {"cor": 2, "outro": 3}
p.desfazer()
assert p.pontos == {"cor": 2, "outro": 0}

# escolher: duas mãos pontuando ou faixa incerta = ambíguo, não pode destravar o placar
assert escolher([(True, 2, "cor")]) == ("cor", 2)
assert escolher([(True, 2, "cor"), (False, 3, "outro")]) == ("cor", 2)
assert escolher([(True, 2, "cor"), (True, 5, "outro")]) == ("cor", 2)  # mão aberta não é ponto
assert escolher([(True, 2, "cor"), (True, 3, "outro")]) == PULAR
assert escolher([(True, 2, None)]) == PULAR
assert escolher([]) is None
assert [quem(f, 0.15) for f in (0.3, 0.1, 0.01)] == ["cor", None, "outro"]

# contar_dedos: mão vertical, pulso em (0,1); dedos esticados com a ponta em y=0, dobrados com y=0.75
def mao(esticados):
    lm = [P(x=0, y=1)] * 21
    for i, (tip, pip) in enumerate([(8, 6), (12, 10), (16, 14), (20, 18)]):
        lm[pip] = P(x=i * 0.1, y=0.6)
        lm[tip] = P(x=i * 0.1, y=0.0 if i < esticados else 0.75)
    return lm

for n in range(5):
    assert contar_dedos(mao(n)) == n, n

# frac_faixa: pulso em (100,100), meio da mão acima -> círculo em (100,116) raio 20
lm = [P(x=0.5, y=0.5)] * 21
lm[9] = P(x=0.5, y=0.3)
for nome, bgr in [("verde", (40, 180, 40)), ("vermelho", (30, 30, 200))]:
    img = np.zeros((200, 200, 3), np.uint8)
    assert frac_faixa(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lm, CORES[nome])[0] == 0
    cv2.circle(img, (100, 116), 20, bgr, -1)
    assert frac_faixa(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lm, CORES[nome])[0] > 0.9, nome
assert frac_faixa(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lm, CORES["verde"])[0] == 0  # vermelho não é verde
print("ok")
