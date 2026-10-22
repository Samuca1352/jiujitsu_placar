from types import SimpleNamespace as P

import cv2
import numpy as np

from placar import (CORES, PULAR, Cronometro, Estado, FonteNavegador, Placar, abrir_fonte, calibrar, classificar,
                    contar_dedos, dono_da_mao, escolher, frac_faixa, Juiz, na_cor, nome_video, quem)

# Placar: segura 0.5s marca uma vez; só marca de novo depois de abaixar
DOIS, TRES, QUATRO = ("cor", "pontos", 2), ("outro", "pontos", 3), ("cor", "pontos", 4)
p = Placar(segurar=0.5, soltar=0.3)
assert [p.update(DOIS, t / 10) for t in range(10)].count(DOIS) == 1
assert p.pontos == {"cor": 2, "outro": 0}
for t in range(10, 15):
    p.update(None, t / 10)
for t in range(15, 25):
    p.update(TRES, t / 10)
assert p.pontos == {"cor": 2, "outro": 3}
p.update(QUATRO, 2.5)  # gesto piscando rápido não pontua
p.update(None, 2.6)
assert p.pontos == {"cor": 2, "outro": 3}
p.desfazer()
assert p.pontos == {"cor": 2, "outro": 0}

# marcações manuais: vantagem/punição entram no histórico e o desfazer volta qualquer uma
p.marcar("outro", "vantagens")
p.marcar("cor", "punicoes")
assert (p.vantagens, p.punicoes) == ({"cor": 0, "outro": 1}, {"cor": 1, "outro": 0})
p.desfazer()
assert p.punicoes["cor"] == 0 and p.vantagens["outro"] == 1
assert [h[4] for h in p.historico] == ["gesto", "manual"]
p2 = Placar(segurar=0.2, soltar=0.2)  # vantagem e punição por gesto entram igual aos pontos
for t in range(5):
    p2.update(("outro", "punicoes", 1), t / 10)
assert p2.punicoes == {"cor": 0, "outro": 1} and p2.pontos == {"cor": 0, "outro": 0}
for ruim in [("ninguem", "pontos"), ("cor", "faltas")]:
    try:
        p.marcar(*ruim)
        raise AssertionError(ruim)
    except ValueError:
        pass

# cronômetro com relógio falso
agora = [0.0]
c = Cronometro(300, relogio=lambda: agora[0])
c.alternar(); agora[0] = 100
assert c.json() == {"restante": 200, "duracao": 300, "rodando": True}
c.alternar(); agora[0] = 250  # pausado não conta
assert c.restante() == 200
c.alternar(); agora[0] = 500
assert c.json()["restante"] == 0 and not c.json()["rodando"]
c.reiniciar(360)
assert c.restante() == 360

# juiz: solto = maior pessoa; travado = segue pela roupa/posição mesmo com alguém maior entrando na frente
def pessoa(x, escala, roupa):
    hist = np.zeros(128, np.float32)
    hist[roupa] = 1
    return P(centro=np.array([x, 0.4]), escala=escala, hist=hist,
             pulsos={"esquerda": np.array([x + escala, 0.6]), "direita": np.array([x - escala, 0.6])})
juiz, atleta = pessoa(0.8, 0.3, 0), pessoa(0.1, 0.2, 50)
j = Juiz()
assert j.escolher([atleta, juiz], 0) == 1
j.travar(juiz, 0)
grande = pessoa(0.5, 0.5, 50)
assert j.escolher([atleta, grande, pessoa(0.85, 0.3, 0)], 0.1) == 2
assert j.escolher([atleta, grande], 0.2) is None  # juiz saiu da imagem: ninguém marca
assert j.escolher([pessoa(1.4, 0.3, 0), atleta], 5) == 0  # voltou em outro lugar: reencontra pela roupa
j.soltar()
assert j.escolher([atleta, grande], 6) == 1

# mão é de quem tem o pulso do Pose mais perto; longe de todo mundo = de ninguém
mao_em = lambda x, y: [P(x=x / (16 / 9), y=y)] * 21
assert dono_da_mao(mao_em(1.1, 0.6), [atleta, juiz], 16 / 9) == (1, "esquerda")
assert dono_da_mao(mao_em(0.5, 0.6), [atleta, juiz], 16 / 9) == (1, "direita")
assert dono_da_mao(mao_em(1.7, 0.1), [atleta, juiz], 16 / 9) == (None, None)

# calibrar: pano laranja-claro (o do teste real: H~15, S~134) com pele parecida na palma
rng = np.random.default_rng(0)
def px(n, h, s, v):
    return np.stack([rng.integers(*h, n), rng.integers(*s, n), rng.integers(*v, n)], 1).astype(np.uint8)
pano, pele = px(400, (12, 18), (110, 175), (130, 240)), px(400, (5, 18), (50, 105), (120, 230))
faixas, css, cobertura, pele_bate = calibrar(np.vstack([pano, pele[:100]]), pele)
assert cobertura > 0.5 and pele_bate <= 0.03, (cobertura, pele_bate)
assert na_cor(pano, faixas).mean() > 0.6
vermelho = np.vstack([px(150, (0, 4), (150, 255), (100, 255)), px(150, (175, 180), (150, 255), (100, 255))])
faixas, *_ = calibrar(vermelho, np.empty((0, 3), np.uint8))
assert len(faixas) == 2  # vermelho dá a volta no 0: duas faixas de matiz
try:
    calibrar(px(300, (0, 180), (0, 30), (0, 255)), pele)  # cinza/branco
    raise AssertionError("calibrou sem cor")
except ValueError:
    pass

# fontes: nomes de vídeo seguros, links validados
assert nome_video("../../etc/luta 1.MP4") == "luta 1.MP4"
for ruim in ["", "x.txt", "../.mp4", ".escondido.mp4"]:
    try:
        nome_video(ruim)
        raise AssertionError(ruim)
    except ValueError:
        pass
for pedido in [{"tipo": "ip", "url": "file:///etc/passwd"}, {"tipo": "ip", "url": ""}, {"tipo": "?"},
               {"tipo": "video", "nome": "nao_existe.mp4"}]:
    try:
        abrir_fonte(pedido, "videos")
        raise AssertionError(pedido)
    except ValueError:
        pass
# vídeo com nome "estranho" copiado à mão pra pasta abre com o nome exato; ../ não sai da pasta
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as pasta:
    gravador = cv2.VideoWriter(str(Path(pasta) / "luta (1), final & cia.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(3):
        gravador.write(np.zeros((48, 64, 3), np.uint8))
    gravador.release()
    assert abrir_fonte({"tipo": "video", "nome": "luta (1), final & cia.mp4"}, pasta).ler()[0].shape == (48, 64, 3)
    for nome in ["../luta (1), final & cia.mp4", "luta _1_, final _ cia.mp4"]:
        try:
            abrir_fonte({"tipo": "video", "nome": nome}, pasta + "/sub")
            raise AssertionError(nome)
        except ValueError:
            pass
f = abrir_fonte({"tipo": "camera"}, "videos")
assert isinstance(f, FonteNavegador)
assert not f.receber(b"lixo")
assert f.receber(cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes())
assert f.ler()[0].shape == (48, 64, 3)

# ajustes ao vivo: limites e validação antes de aplicar
est = Estado(Placar(), {"cor": "verde", "nomes": {"cor": "A", "outro": "B"}, "min_frac": 0.15}, "videos")
est.configurar({"min_frac": 5, "segurar": 1.2, "cor": "azul", "nomes": {"cor": " Ana ", "outro": "Bia"}})
assert est.cfg["min_frac"] == 0.9 and est.placar.segurar == 1.2 and est.cfg["cor"] == "azul" and est.cfg["nomes"]["cor"] == "Ana"
try:
    est.configurar({"min_frac": 0.3, "cor": "roxo"})
    raise AssertionError("cor inválida passou")
except ValueError:
    assert est.cfg["min_frac"] == 0.9  # nada aplicado

# escolher: duas mãos sinalizando ou faixa incerta = ambíguo, não pode destravar o placar
PTS2, PTS3, VANT, PUN = ("pontos", 2), ("pontos", 3), ("vantagens", 1), ("punicoes", 1)
assert escolher([(PTS2, "cor")]) == ("cor", "pontos", 2)
assert escolher([(PTS2, "cor"), (None, "outro")]) == ("cor", "pontos", 2)
assert escolher([(VANT, "outro")]) == ("outro", "vantagens", 1)
assert escolher([(PTS2, "cor"), (PTS3, "outro")]) == PULAR
assert escolher([(PUN, None)]) == PULAR
assert escolher([]) is None
assert [quem(f, 0.15) for f in (0.3, 0.1, 0.01)] == ["cor", None, "outro"]

# contar_dedos: mão vertical, pulso em (0,1); dedos esticados com a ponta em y=0, dobrados com y=0.75
def mao(esticados):
    lm = [P(x=0, y=1)] * 21
    lm[9] = P(x=0.15, y=0.7)  # base do dedo médio: dá a direção da mão
    for i, (tip, pip) in enumerate([(8, 6), (12, 10), (16, 14), (20, 18)]):
        lm[pip] = P(x=i * 0.1, y=0.6)
        lm[tip] = P(x=i * 0.1, y=0.0 if i < esticados else 0.75)
    return lm

for n in range(5):
    assert contar_dedos(mao(n)) == n, n

# classificar: mão() fica com o pulso em y=1; desloca pra cima/baixo e deita girando 90°
def mover(lm, dy=0.0, deitar=False):
    return [P(x=0.5 + (1 - l.y) * 0.3 if deitar else 0.5 + l.x * 0.3, y=(0.6 + l.x * 0.05 if deitar else l.y * 0.3 + 0.3) + dy)
            for l in lm]

alta, ombro, baixa = -0.3, 0.06, 0.4  # pulso em 0.3, 0.66 e 1.0 com os ombros em 0.6
assert classificar(mover(mao(2), alta), 0.6) == ("pontos", 2)
assert classificar(mover(mao(4), alta), 0.6) == ("pontos", 4)  # 4 dedos de pé = 4 pontos, não vantagem
assert classificar(mover(mao(2), ombro), 0.6) is None  # pontos exigem o pulso acima dos ombros
assert classificar(mover(mao(0), ombro), 0.6) == ("punicoes", 1)  # punho fechado na altura do ombro
assert classificar(mover(mao(4), 0, deitar=True), 0.6) == ("vantagens", 1)  # espalmada na horizontal
assert classificar(mover(mao(0), baixa), 0.6) is None  # mão fechada abaixada = descansando
assert classificar(mover(mao(4), baixa, deitar=True), 0.6) is None
assert classificar(mover(mao(1), alta), 0.6) is None

# punho de verdade x mão relaxada curvada (medidas 3D reais: pontas a ~0.8 e ~1.15 palmas do pulso)
def mundo(dist):
    lm = [P(x=0, y=0, z=0)] * 21
    lm[9] = P(x=0, y=-1, z=0)
    for tip, _ in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lm[tip] = P(x=0, y=-dist * 0.6, z=dist * 0.8)
    return lm
assert classificar(mover(mao(0), ombro), 0.6, mundo=mundo(0.8)) == ("punicoes", 1)
assert classificar(mover(mao(0), ombro), 0.6, mundo=mundo(1.15)) is None

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
