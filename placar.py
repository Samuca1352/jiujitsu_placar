"""Placar de jiu-jitsu por gesto do juiz (sinais da IBJJF).

Só valem as mãos do juiz: o modelo de pose acha as pessoas e segue o juiz (a maior pessoa, ou a que foi travada no painel).
  pulso acima dos ombros dele com 2, 3 ou 4 dedos -> pontos;
  na altura do ombro: mão espalmada na horizontal -> vantagem · punho fechado -> punição.
De quem é o sinal (IBJJF: o juiz usa sempre a mesma mão pro mesmo atleta):
  modo "mao":   braço direito do juiz -> atleta "cor" (faixa verde-amarela); esquerdo -> o outro;
  modo "faixa": mão com a faixa colorida no pulso -> atleta "cor"; mão sem faixa -> o outro.
Painel web em http://localhost:8000: escolhe a fonte (vídeo, câmera do PC ou câmera IP), mostra o vídeo
com as caixas, placar com vantagens/punições, cronômetro, ajustes ao vivo, desfazer/zerar.
"""
import argparse
import json
import re
import sys
import threading
import time
import webbrowser
from functools import cached_property
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions, PoseLandmarker, PoseLandmarkerOptions,
                                           RunningMode)

# HSV do OpenCV: H 0-179. Vermelho dá a volta no 0, por isso duas faixas.
CORES = {
    "verde": [((40, 80, 60), (85, 255, 255))],
    "amarelo": [((20, 100, 100), (35, 255, 255))],
    "azul": [((95, 120, 60), (130, 255, 255))],
    "vermelho": [((0, 120, 80), (8, 255, 255)), ((170, 120, 80), (179, 255, 255))],
    "laranja": [((9, 130, 120), (19, 255, 255))],
    "rosa": [((140, 80, 100), (169, 255, 255))],
}
CSS = {"verde": "#22c55e", "amarelo": "#eab308", "azul": "#3b82f6", "vermelho": "#ef4444", "laranja": "#f97316", "rosa": "#ec4899"}
PONTOS_VALIDOS = {2, 3, 4}  # IBJJF: 2 queda/raspagem/joelho, 3 passagem, 4 montada/costas
DEDOS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (ponta, articulação PIP) indicador..mindinho
PULAR = "pular"  # frame ambíguo: não mexe no placar
FOLGA_OMBRO = 0.35  # vantagem/punição: centro da mão até 0.35 "escala do corpo" abaixo da linha dos ombros
POSE_A_CADA = 2  # o corpo roda em quadros alternados: o gesto dura 0.5s, o juiz não muda de lugar em 1/30s
MAX_LARGURA = 960  # os modelos reduzem a imagem de qualquer jeito; reduzir antes deixa a conversão mais leve
# landmarks do Pose (lados do próprio corpo da pessoa)
OMBRO_E, OMBRO_D, PULSO_E, PULSO_D, QUADRIL_E, QUADRIL_D = 11, 12, 15, 16, 23, 24
TIPOS = ("pontos", "vantagens", "punicoes")
ROTULO = {"pontos": "+{}", "vantagens": "VANTAGEM", "punicoes": "PUNICAO"}  # putText do OpenCV não tem acento
LIMITES = {"min_frac": (0.01, 0.9), "segurar": (0.1, 3.0), "soltar": (0.1, 3.0), "fps": (5, 60)}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
# pasta do painel e dos modelos: ao lado do placar.py, ou a pasta temporária do executável (PyInstaller)
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
ESQUEMAS_IP = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://")


def contar_dedos(lm):
    """Dedos esticados (sem polegar): ponta mais longe do pulso que a PIP. Não depende da rotação da mão."""
    p = np.array([(l.x, l.y) for l in lm])
    d = lambda i: np.linalg.norm(p[i] - p[0])
    return int(sum(d(tip) > d(pip) * 1.1 for tip, pip in DEDOS))


def circulo_pulso(lm, w, h):
    """Círculo logo abaixo do pulso, no sentido do antebraço: onde a faixa deve estar. Retorna (centro, raio)."""
    pulso = np.array([lm[0].x * w, lm[0].y * h])
    meio = np.array([lm[9].x * w, lm[9].y * h])
    cx, cy = (pulso + (pulso - meio) * 0.4).astype(int)
    return (int(cx), int(cy)), max(int(np.linalg.norm(meio - pulso) * 0.5), 6)


def pixels_pulso(hsv, lm):
    """Pixels HSV (N x 3) dentro do círculo do pulso."""
    h, w = hsv.shape[:2]
    centro, raio = circulo_pulso(lm, w, h)
    roi = np.zeros((h, w), np.uint8)
    cv2.circle(roi, centro, raio, 255, -1)
    return hsv[roi > 0]


def pixels_palma(hsv, lm):
    """Pixels HSV (N x 3) da palma: polígono pulso -> bases dos dedos. Serve de amostra da pele."""
    h, w = hsv.shape[:2]
    roi = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi, [np.array([(lm[i].x * w, lm[i].y * h) for i in (0, 1, 5, 9, 13, 17)], np.int32)], 255)
    return hsv[roi > 0]


def na_cor(px, faixas):
    """Máscara booleana dos pixels (N x 3) dentro de alguma das faixas HSV."""
    m = np.zeros(len(px), bool)
    for lo, hi in faixas:
        m |= np.all((px >= lo) & (px <= hi), axis=1)
    return m


def frac_faixa(hsv, lm, faixas):
    """Fração da cor no círculo do pulso. Retorna (frac, centro, raio)."""
    h, w = hsv.shape[:2]
    centro, raio = circulo_pulso(lm, w, h)
    px = pixels_pulso(hsv, lm)
    return (float(na_cor(px, faixas).mean()) if len(px) else 0.0), centro, raio


def calibrar(pulso, pele):
    """Aprende a cor da faixa a partir dos pixels do círculo do pulso (N x 3 HSV) sem pegar a pele (pixels da palma).
    Retorna (faixas HSV, cor css, fração do pulso que bate, fração da pele que bate). ValueError se não der."""
    faixa = pulso[pulso[:, 1] >= 50]  # só o que tem cor de verdade; branco/cinza/preto não serve de faixa
    if len(faixa) < 30 or len(faixa) < len(pulso) * 0.15:
        raise ValueError("não vi cor forte no pulso: ponha a faixa dentro do círculo e tente de novo")
    # matiz é circular (vermelho fica em 0 e 179): gira pra média cair em 90 antes de tirar percentis
    ang = faixa[:, 0] * np.pi / 90
    meio = int(round(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 90 / np.pi)) % 180
    lo, hi = (np.percentile((faixa[:, 0].astype(int) - meio + 90) % 180, [10, 90]) + (-4, 4)).astype(int) + meio - 90
    matizes = [(lo, hi)] if 0 <= lo and hi <= 179 else [(0, hi % 180), (lo % 180, 179)]
    s_lo = int(np.percentile(faixa[:, 1], 15) * 0.8)
    v_lo = int(np.percentile(faixa[:, 2], 10) * 0.6)
    faixas = lambda: [((a, s_lo, v_lo), (b, 255, 255)) for a, b in matizes]
    # pele parecida (laranja, rosa, vermelho): sobe a saturação mínima até a palma quase não bater
    while len(pele) and na_cor(pele, faixas()).mean() > 0.03 and s_lo < 230:
        s_lo += 10
    bate = na_cor(pulso, faixas())
    if bate.mean() < 0.1:
        raise ValueError("a cor da faixa é parecida demais com a pele: use uma faixa de cor mais forte (verde, azul)")
    bgr = cv2.cvtColor(np.uint8([[np.median(pulso[bate], 0)]]), cv2.COLOR_HSV2BGR)[0, 0]
    pele_bate = float(na_cor(pele, faixas()).mean()) if len(pele) else 0.0
    return faixas(), "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0]), float(bate.mean()), pele_bate


def quem(frac, min_frac):
    """Faixa clara -> "cor"; quase nada da cor -> "outro"; meio-termo (faixa meio escondida) -> None."""
    return "cor" if frac >= min_frac else "outro" if frac < min_frac / 3 else None


def punho_fechado(mundo):
    """Punho de verdade: pontas dos dedos perto do pulso (média < 1 palma). Mão só relaxada/curvada fica em ~1.1-1.3.
    Usa os landmarks 3D (hand_world_landmarks), que não dependem do ângulo da câmera."""
    p = np.array([(l.x, l.y, l.z) for l in mundo])
    palma = np.linalg.norm(p[9] - p[0])
    return bool(np.mean([np.linalg.norm(p[t] - p[0]) for t, _ in DEDOS]) < palma)


def horizontal(lm, aspecto):
    """Mão deitada: o eixo pulso -> base do dedo médio aponta mais pro lado que pra cima. `aspecto` = largura/altura."""
    dx, dy = (lm[9].x - lm[0].x) * aspecto, lm[9].y - lm[0].y
    return abs(dx) > 1.5 * abs(dy)


def classificar(lm, ombro, folga=0.1, aspecto=16 / 9, mundo=None):
    """Sinal do juiz numa mão: ("pontos", 2|3|4), ("vantagens", 1), ("punicoes", 1) ou None.
    `ombro` = altura (0=topo, 1=base) da linha dos ombros do juiz; `folga` = quanto abaixo dela ainda é "no ombro".
    - pontos: pulso acima dos ombros com 2, 3 ou 4 dedos (de pé);
    - vantagem: braço esticado na horizontal, mão espalmada (4 dedos esticados, mão deitada);
    - punição: braço levantado na altura do ombro com o punho fechado (nenhum dedo; com `mundo`, confere o punho em 3D)."""
    dedos, y = contar_dedos(lm), lm[0].y
    no_ombro = lm[9].y < ombro + folga  # centro da mão: no punho fechado o pulso fica bem abaixo dele
    if dedos == 4 and no_ombro and horizontal(lm, aspecto):
        return "vantagens", 1
    if dedos == 0 and no_ombro and (mundo is None or punho_fechado(mundo)):
        return "punicoes", 1
    if dedos in PONTOS_VALIDOS and y < ombro:
        return "pontos", dedos
    return None


class Pessoa:
    """Uma pessoa do Pose: medidas em coordenadas onde x já foi multiplicado pelo aspecto (distâncias iguais nos 2 eixos)."""

    def __init__(self, pose, aspecto, hsv):
        self.pose, self.aspecto, self.hsv = pose, aspecto, hsv
        p = lambda i: np.array([pose[i].x * aspecto, pose[i].y])
        self.centro = (p(OMBRO_E) + p(OMBRO_D)) / 2
        self.ombro = float(self.centro[1])
        # escala do corpo: largura dos ombros, ou 60% do tronco se o quadril aparece (juiz de lado tem ombro "estreito")
        self.escala = float(np.linalg.norm(p(OMBRO_E) - p(OMBRO_D)))
        if min(pose[QUADRIL_E].visibility, pose[QUADRIL_D].visibility) > 0.5:
            self.escala = max(self.escala, 0.6 * float(np.linalg.norm((p(QUADRIL_E) + p(QUADRIL_D)) / 2 - self.centro)))
        self.escala = max(self.escala, 0.02)
        self.pulsos = {"esquerda": p(PULSO_E), "direita": p(PULSO_D)}

    @cached_property
    def hist(self):
        """Só é calculado quando o juiz está travado (ou na hora de travar): custa ~1 ms por pessoa."""
        return roupa(self.hsv, self)


def roupa(hsv, pessoa):
    """Histograma HSV do tronco: a "cara" do juiz pra reencontrar ele (juiz IBJJF de preto, atletas de kimono)."""
    h, w = hsv.shape[:2]
    pose = pessoa.pose
    ombros = [(pose[i].x * w, pose[i].y * h) for i in (OMBRO_E, OMBRO_D)]
    if min(pose[QUADRIL_E].visibility, pose[QUADRIL_D].visibility) > 0.5:
        baixo = [(pose[i].x * w, pose[i].y * h) for i in (QUADRIL_D, QUADRIL_E)]
    else:  # sem quadril na imagem: um pedaço do peito abaixo dos ombros
        desce = pessoa.escala * h
        baixo = [(x, min(y + desce, h - 1)) for x, y in reversed(ombros)]
    roi = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi, [np.array(ombros + baixo, np.int32)], 255)
    hist = cv2.calcHist([hsv], [0, 1, 2], roi, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


class Juiz:
    """Escolhe qual pessoa é o juiz a cada quadro.
    Solto: a maior pessoa da imagem (a câmera fica de frente pro juiz, mais perto dele que dos outros).
    Travado: segue pela posição (quadro a quadro) e confere pela roupa; se sumir, reencontra pela roupa."""

    def __init__(self):
        self.soltar()

    def soltar(self):
        self.travado, self.hist, self.centro, self.escala, self.visto = False, None, None, 1.0, -1e9

    def travar(self, pessoa, t):
        self.travado, self.hist = True, pessoa.hist
        self._viu(pessoa, t)

    def _viu(self, pessoa, t):
        self.centro, self.escala, self.visto = pessoa.centro, pessoa.escala, t

    def escolher(self, pessoas, t):
        """Índice do juiz em `pessoas`, ou None (juiz fora da imagem)."""
        if not pessoas:
            return None
        if not self.travado:
            i = max(range(len(pessoas)), key=lambda i: pessoas[i].escala)
            self._viu(pessoas[i], t)
            return i
        recente = t - self.visto < 1.0

        def custo(p):
            d_roupa = cv2.compareHist(self.hist, p.hist, cv2.HISTCMP_BHATTACHARYYA)
            d_pos = float(np.linalg.norm(p.centro - self.centro)) / self.escala if recente else 0.0
            return d_roupa + 0.5 * d_pos, d_roupa, d_pos

        i = min(range(len(pessoas)), key=lambda i: custo(pessoas[i])[0])
        _, d_roupa, d_pos = custo(pessoas[i])
        if d_roupa > 0.6 and not (recente and d_pos < 0.5):
            return None  # ninguém parece o juiz
        self._viu(pessoas[i], t)
        if d_roupa < 0.35:  # roupa muda com a luz: aprende devagar
            self.hist = (0.95 * self.hist + 0.05 * pessoas[i].hist).astype(np.float32)
        return i


def dono_da_mao(lm, pessoas, aspecto):
    """(índice da pessoa, "esquerda"|"direita") cujo pulso do Pose está mais perto do pulso da mão, ou (None, None)."""
    pulso = np.array([lm[0].x * aspecto, lm[0].y])
    melhor = min(((float(np.linalg.norm(p.pulsos[lado] - pulso)) / p.escala, i, lado)
                  for i, p in enumerate(pessoas) for lado in p.pulsos), default=None)
    if melhor is None or melhor[0] > 0.6:
        return None, None
    return melhor[1], melhor[2]


def escolher(maos):
    """maos = [(sinal, quem)]. Retorna (quem, tipo, n), None (ninguém sinalizando) ou PULAR (ambíguo)."""
    g = [(q, *s) for s, q in maos if s]
    if len(g) > 1 or any(q is None for q, *_ in g):
        return PULAR
    return g[0] if g else None


class Placar:
    """Gesto precisa ficar parado `segurar` segundos; depois só marca de novo quando a mão abaixa por `soltar`.
    Pelo painel também dá pra marcar à mão (pontos, vantagem, punição)."""

    def __init__(self, segurar=0.5, soltar=0.8):
        self.segurar, self.soltar = segurar, soltar
        self.zerar()

    def zerar(self):
        self.pontos, self.vantagens, self.punicoes = ({"cor": 0, "outro": 0} for _ in TIPOS)
        self.historico = []  # (quem, tipo, n, hora unix, "gesto"|"manual")
        self.soltar_gesto()

    def soltar_gesto(self):
        """Esquece o gesto em andamento (ex.: trocou a fonte no meio de um gesto)."""
        self.candidato, self.desde, self.travado, self.vazio_desde = None, 0.0, False, None

    def marcar(self, q, tipo="pontos", n=1, origem="manual"):
        if q not in ("cor", "outro") or tipo not in TIPOS:
            raise ValueError("atleta ou tipo inválido")
        getattr(self, tipo)[q] += n
        self.historico.append((q, tipo, n, round(time.time(), 1), origem))

    def update(self, gesto, t):
        """gesto = ("cor"|"outro", tipo, n) ou None. Retorna o gesto quando marca."""
        if gesto is None:
            self.vazio_desde = self.vazio_desde if self.vazio_desde is not None else t
            if t - self.vazio_desde >= self.soltar:
                self.candidato, self.travado = None, False
            return None
        self.vazio_desde = None
        if gesto != self.candidato:
            self.candidato, self.desde = gesto, t
            return None
        if not self.travado and t - self.desde >= self.segurar:
            self.travado = True
            self.marcar(*gesto, origem="gesto")
            return gesto
        return None

    def desfazer(self):
        if self.historico:
            q, tipo, n, *_ = self.historico.pop()
            getattr(self, tipo)[q] -= n

    def json(self):
        return {"pontos": self.pontos, "vantagens": self.vantagens, "punicoes": self.punicoes, "historico": self.historico}


class Cronometro:
    """Tempo da luta, contando pra baixo. `relogio` é injetável pros testes."""

    def __init__(self, duracao=300, relogio=time.monotonic):
        self.relogio, self.duracao = relogio, duracao
        self.reiniciar()

    def reiniciar(self, duracao=None):
        self.duracao = duracao or self.duracao
        self.gasto, self.desde = 0.0, None

    def restante(self):
        correndo = self.relogio() - self.desde if self.desde is not None else 0
        return max(self.duracao - self.gasto - correndo, 0.0)

    def alternar(self):
        if self.desde is not None:
            self.gasto += self.relogio() - self.desde
            self.desde = None
        elif self.restante() > 0:
            self.desde = self.relogio()

    def json(self):
        r = self.restante()
        return {"restante": round(r, 1), "duracao": self.duracao, "rodando": self.desde is not None and r > 0}


# ---------------------------------------------------------------- fontes de vídeo
# Toda fonte tem ler() -> (frame | None, segundos desde o início), `acabou` e fechar().
# ler() de fonte ao vivo devolve None depois de 1s sem quadro, pra quem lê poder notar troca de fonte.

class FonteVideo:
    """Arquivo de vídeo. Com painel toca em tempo real; sem painel corre o mais rápido possível."""
    tipo = "video"

    def __init__(self, caminho, tempo_real=True):
        self.nome, self.tempo_real, self.inicio, self.acabou = Path(caminho).name, tempo_real, None, False
        self.cap = cv2.VideoCapture(str(caminho))
        if not self.cap.isOpened():
            raise ValueError(f"não consegui abrir o vídeo {self.nome}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

    def ler(self):
        ok, frame = self.cap.read()
        if not ok:
            self.acabou = True
            return None, 0.0
        t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if self.tempo_real:
            self.inicio = self.inicio if self.inicio is not None else time.monotonic() - t
            atraso = time.monotonic() - self.inicio - t
            if atraso > 2 / self.fps:  # máquina lenta: pula quadros pra continuar em tempo real
                for _ in range(int(atraso * self.fps)):
                    self.cap.grab()
                return self.ler()
            time.sleep(max(0.0, -atraso))
        return frame, t

    def fechar(self):
        pass  # o cap é solto pelo GC; soltar aqui pode cruzar com um read() da thread de vídeo


class _AoVivo:
    """Guarda só o quadro mais novo, senão o buffer vira atraso."""
    acabou = False

    def __init__(self):
        self.frame, self.novo, self.inicio = None, threading.Event(), time.monotonic()

    def _chegou(self, frame):
        self.frame = frame
        self.novo.set()

    def ler(self):
        if not self.novo.wait(1):
            return None, 0.0
        self.novo.clear()
        return self.frame, time.monotonic() - self.inicio

    def fechar(self):
        pass


class FonteIP(_AoVivo):
    """Câmera IP (rtsp/http, ex.: app IP Webcam no celular) ou índice de câmera local ("0").
    Uma thread lê sem parar; se o stream cair, tenta reconectar."""
    tipo = "ip"

    def __init__(self, url):
        super().__init__()
        self.nome, self.ligada = url, True
        self.cap = self._abrir()
        if not self.cap.isOpened():
            raise ValueError(f"não conectou em {url}. Teste o link no navegador e confira se está na mesma rede.")
        threading.Thread(target=self._ler, daemon=True).start()

    def _abrir(self):
        if self.nome.isdigit():
            return cv2.VideoCapture(int(self.nome))
        # sem timeout o OpenCV fica até 30s pendurado num IP errado
        return cv2.VideoCapture(self.nome, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])

    def _ler(self):
        while self.ligada:
            ok, frame = self.cap.read()
            if ok:
                self._chegou(frame)
                continue
            self.cap.release()
            time.sleep(1)
            if self.ligada:
                self.cap = self._abrir()
        self.cap.release()

    def fechar(self):
        self.ligada = False


class FonteNavegador(_AoVivo):
    """Câmera do PC: o painel pega a webcam no navegador e manda cada quadro em JPEG (POST /quadro).
    Funciona até com o servidor no Docker/WSL, que não enxerga a webcam."""
    tipo, nome = "camera", "câmera do navegador"

    def receber(self, jpg):
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False
        self._chegou(frame)
        return True


def nome_video(nome):
    """Nome seguro pra gravar um vídeo enviado, ou ValueError. (Pra abrir, vale o nome como está na pasta.)"""
    nome = re.sub(r"[^\w.\- ]", "_", Path(nome or "").name).strip()
    if not nome or nome.startswith(".") or Path(nome).suffix.lower() not in VIDEO_EXT:
        raise ValueError(f"vídeo inválido: use {', '.join(sorted(VIDEO_EXT))}")
    return nome


def listar_videos(pasta):
    return sorted(p.name for p in Path(pasta).glob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def abrir_fonte(pedido, pasta_videos, tempo_real=True):
    """pedido = {"tipo": "video", "nome": ...} | {"tipo": "camera"} | {"tipo": "ip", "url": ...}"""
    tipo = pedido.get("tipo")
    if tipo == "video":
        nome = Path(str(pedido.get("nome") or "")).name  # só o nome: nada de ../ pra fora da pasta
        if nome not in listar_videos(pasta_videos):
            raise ValueError(f"vídeo não encontrado: {nome}")
        return FonteVideo(Path(pasta_videos) / nome, tempo_real)
    if tipo == "camera":
        return FonteNavegador()
    if tipo == "ip":
        url = str(pedido.get("url", "")).strip()
        if not (url.isdigit() or url.lower().startswith(ESQUEMAS_IP)):
            raise ValueError("link inválido: use http://IP:porta/video, rtsp://... ou o número de uma câmera local")
        return FonteIP(url)
    raise ValueError(f"tipo de fonte desconhecido: {tipo}")


def fonte_da_linha(fonte, pasta_videos):
    """--fonte da linha de comando: número = câmera local, com :// = câmera IP, senão arquivo de vídeo.
    Retorna (pedido pra abrir_fonte, pasta do vídeo)."""
    if fonte.isdigit() or "://" in fonte:
        return {"tipo": "ip", "url": fonte}, pasta_videos
    return {"tipo": "video", "nome": Path(fonte).name}, Path(fonte).parent


# ---------------------------------------------------------------- estado compartilhado

class Estado:
    """O que a thread de vídeo publica e o painel lê/muda. `cond` protege tudo."""

    def __init__(self, placar, cfg, pasta_videos):
        self.cond, self.placar, self.cfg, self.pasta_videos = threading.Condition(), placar, cfg, Path(pasta_videos)
        self.relogio = Cronometro()
        self.fonte, self.status = None, "escolha uma fonte"
        self.jpg, self.n, self.maos, self.fps = None, 0, [], 0.0
        self.cores, self.css = dict(CORES), dict(CSS)
        self.amostra = None  # (hsv, [landmarks das mãos do juiz]) do último quadro, pra calibrar a cor da faixa
        self.juiz, self.pessoas, self.i_juiz = Juiz(), [], None  # pessoas/i_juiz: do último quadro

    def trocar_fonte(self, nova):
        with self.cond:
            if self.fonte:
                self.fonte.fechar()
            self.fonte, self.status, self.maos = nova, "abrindo", []
            self.placar.soltar_gesto()
            self.cond.notify_all()

    def configurar(self, novo):
        """Muda ajustes ao vivo. Valida tudo antes de aplicar qualquer coisa."""
        cfg = {}
        if "cor" in novo:
            if novo["cor"] not in self.cores:
                raise ValueError("cor inválida")
            cfg["cor"] = novo["cor"]
        if "nomes" in novo:
            nomes = {k: str(novo["nomes"].get(k, "")).strip()[:24] for k in ("cor", "outro")}
            if not all(nomes.values()):
                raise ValueError("os dois atletas precisam de nome")
            cfg["nomes"] = nomes
        if "modo" in novo:
            if novo["modo"] not in ("mao", "faixa"):
                raise ValueError("modo inválido")
            cfg["modo"] = novo["modo"]
        if "inverter" in novo:
            cfg["inverter"] = bool(novo["inverter"])
        for k, (lo, hi) in LIMITES.items():
            if k in novo:
                cfg[k] = min(max(float(novo[k]), lo), hi)
        with self.cond:
            self.placar.segurar = cfg.pop("segurar", self.placar.segurar)
            self.placar.soltar = cfg.pop("soltar", self.placar.soltar)
            self.cfg.update(cfg)

    def travar_juiz(self, x=None, y=None):
        """Trava o juiz: a pessoa mais perto do clique (x, y de 0 a 1 na imagem) ou, sem clique, a escolhida agora."""
        with self.cond:
            if not self.pessoas:
                raise ValueError("ninguém na imagem")
            if x is None:
                if self.i_juiz is None:
                    raise ValueError("clique no juiz no vídeo pra escolher")
                i = self.i_juiz
            else:
                a = self.pessoas[0].aspecto
                i = min(range(len(self.pessoas)),
                        key=lambda i: np.linalg.norm(self.pessoas[i].centro - (float(x) * a, float(y))) / self.pessoas[i].escala)
            self.juiz.travar(self.pessoas[i], time.monotonic())
            self.i_juiz = i

    def calibrar(self):
        """Aprende a cor da faixa na mão do último quadro que tem mais cor no pulso. Retorna o relatório."""
        with self.cond:
            if not self.amostra or not self.amostra[1]:
                raise ValueError("nenhuma mão na imagem: levante a mão com a faixa no pulso")
            hsv, maos = self.amostra
        amostras = [(pixels_pulso(hsv, lm), pixels_palma(hsv, lm)) for lm in maos]
        pulso, palma = max(amostras, key=lambda a: np.median(a[0][:, 1]) if len(a[0]) else 0)
        faixas, css, cobertura, pele = calibrar(pulso, palma)
        with self.cond:
            self.cores["calibrada"], self.css["calibrada"] = faixas, css
            self.cfg.update(cor="calibrada", modo="faixa", min_frac=round(min(max(cobertura * 2 / 3, 0.05), 0.9), 2))
        return {"cobertura": round(cobertura, 2), "pele": round(pele, 2), "css": css}

    def json(self):
        with self.cond:
            f = self.fonte
            return json.dumps({
                **self.placar.json(),
                "cfg": {**self.cfg, "css": self.css[self.cfg["cor"]], "segurar": self.placar.segurar, "soltar": self.placar.soltar},
                "cores": self.css,
                "fonte": {"tipo": f.tipo, "nome": f.nome} if f else None,
                "status": self.status, "maos": self.maos, "fps": self.fps,
                "relogio": self.relogio.json(),
                "juiz": {"travado": self.juiz.travado, "visivel": self.i_juiz is not None, "pessoas": len(self.pessoas)},
            })


# ---------------------------------------------------------------- processamento

def desenhar_mao(frame, lm, cor_bgr, texto, grossa):
    """Caixa estilo YOLO em volta da mão, com etiqueta preenchida em cima."""
    h, w = frame.shape[:2]
    xs, ys = [l.x * w for l in lm], [l.y * h for l in lm]
    x0, y0, x1, y1 = int(min(xs)) - 15, int(min(ys)) - 15, int(max(xs)) + 15, int(max(ys)) + 15
    cv2.rectangle(frame, (x0, y0), (x1, y1), cor_bgr, 3 if grossa else 1)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x0, y0 - th - 10), (x0 + tw + 10, y0), cor_bgr, -1)
    cv2.putText(frame, texto, (x0 + 5, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    for l in lm:
        cv2.circle(frame, (int(l.x * w), int(l.y * h)), 2, cor_bgr, -1)


def desenhar_pessoa(frame, pessoa, cor_bgr, texto, grossa):
    """Caixa em volta dos pontos visíveis do corpo."""
    h, w = frame.shape[:2]
    pts = np.array([(l.x * w, l.y * h) for l in pessoa.pose if l.visibility > 0.5] or [(0, 0)])
    (x0, y0), (x1, y1) = pts.min(0).astype(int), pts.max(0).astype(int)
    cv2.rectangle(frame, (x0, y0), (x1, y1), cor_bgr, 2 if grossa else 1)
    cv2.putText(frame, texto, (x0 + 4, max(y0 + 18, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_bgr, 2 if grossa else 1)
    return x0, x1


def processar(est, saida=None, painel=True):
    """Lê a fonte atual, acha o juiz e as mãos dele e atualiza o placar. Sem painel, sai quando o vídeo acaba.
    Analisa no máximo cfg["fps"] quadros por segundo (do tempo da fonte): o gesto dura 0.5s, então 15/s sobram
    e a CPU cai pela metade com câmera de 30 fps. Muda ao vivo pelo painel."""
    placar = est.placar
    det_mao = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(BASE / "hand_landmarker.task")), running_mode=RunningMode.VIDEO, num_hands=4))
    det_pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(BASE / "pose_landmarker_lite.task")), running_mode=RunningMode.VIDEO, num_poses=4))
    fonte, gravador, ts, dt, antes, n, corpos, t_ultimo = None, None, 0, 1.0, time.monotonic(), 0, [], 0.0

    while True:
        with est.cond:
            if est.fonte is None or (est.fonte is fonte and fonte.acabou):
                if not painel:
                    break
                est.cond.wait(1)
                continue
            fonte, cfg = est.fonte, {**est.cfg, "nomes": dict(est.cfg["nomes"])}
            faixas_cor, css_cor = est.cores[cfg["cor"]], est.css[cfg["cor"]]
        frame, t = fonte.ler()
        if frame is None:
            with est.cond:
                if est.fonte is fonte:
                    est.status, est.fps = ("fim do vídeo" if fonte.acabou else "sem sinal"), 0.0
            continue
        if 0 <= t - t_ultimo < 0.8 / cfg["fps"]:  # (t volta pra 0 ao trocar de fonte: aí analisa)
            continue
        t_ultimo = t

        h, w = frame.shape[:2]
        if w > MAX_LARGURA:
            frame = cv2.resize(frame, (MAX_LARGURA, round(h * MAX_LARGURA / w)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        aspecto = w / h
        nomes, cor_bgr = cfg["nomes"], tuple(int(css_cor[i:i + 2], 16) for i in (5, 3, 1))
        ts = max(int(time.monotonic() * 1000), ts + 1)  # os detectores exigem timestamp sempre crescente, mesmo trocando de fonte
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res, n = det_mao.detect_for_video(img, ts), n + 1
        if n % POSE_A_CADA == 1 or POSE_A_CADA == 1:
            corpos = det_pose.detect_for_video(img, ts).pose_landmarks
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        pessoas = [Pessoa(pose, aspecto, hsv) for pose in corpos]
        with est.cond:
            i_juiz = est.juiz.escolher(pessoas, time.monotonic())
        juiz = pessoas[i_juiz] if i_juiz is not None else None
        for i, p in enumerate(pessoas):
            x0, x1 = desenhar_pessoa(frame, p, (255, 80, 200) if i == i_juiz else (110, 110, 110),
                                     ("JUIZ (travado)" if est.juiz.travado else "JUIZ") if i == i_juiz else "", i == i_juiz)
            if i == i_juiz:  # linha dos ombros do juiz: pulso acima dela = pontos
                cv2.line(frame, (x0, int(p.ombro * h)), (x1, int(p.ombro * h)), (0, 255, 255), 1)

        maos, info_maos, lms_juiz = [], [], []
        for lm, mundo in zip(res.hand_landmarks, res.hand_world_landmarks):
            dono, lado = dono_da_mao(lm, pessoas, aspecto)
            dedos = contar_dedos(lm)
            if juiz is None or dono != i_juiz:
                desenhar_mao(frame, lm, (90, 90, 90), "ignorada", False)
                continue
            if cfg["inverter"]:
                lado = {"direita": "esquerda", "esquerda": "direita"}[lado]
            lms_juiz.append(lm)
            sinal = classificar(lm, juiz.ombro, FOLGA_OMBRO * juiz.escala, aspecto, mundo)
            if cfg["modo"] == "mao":
                frac, q, extra = 0.0, {"direita": "cor", "esquerda": "outro"}[lado], lado[:3]
            else:
                frac, centro, raio = frac_faixa(hsv, lm, faixas_cor)
                q, extra = quem(frac, cfg["min_frac"]), f"{frac:.0%}"
            maos.append((sinal, q))
            info_maos.append({"dedos": dedos, "faixa": round(frac, 2), "lado": lado, "sinal": sinal, "quem": q})
            caixa = cor_bgr if q == "cor" else (235, 235, 235) if q == "outro" else (0, 165, 255)
            rotulo = (f"{ROTULO[sinal[0]].format(sinal[1])} {nomes[q]}" if sinal and q
                      else f"{dedos} dedos" + ("" if q else " faixa?"))
            desenhar_mao(frame, lm, caixa if sinal else (140, 140, 140), f"{rotulo} {extra}", bool(sinal))
            if cfg["modo"] == "faixa":
                cv2.circle(frame, centro, raio, caixa, 1)

        gesto = escolher(maos)
        if saida:
            gravador = gravador or cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), getattr(fonte, "fps", 30), (w, h))
            gravador.write(frame)
        agora = time.monotonic()
        dt, antes = 0.9 * dt + 0.1 * (agora - antes), agora
        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes() if painel else None

        with est.cond:
            if est.fonte is not fonte:  # trocaram a fonte enquanto este quadro era processado
                continue
            ev = None if gesto == PULAR else placar.update(gesto, t)
            est.jpg, est.n, est.maos, est.fps = jpg, est.n + 1, info_maos, round(1 / max(dt, 1e-3), 1)
            est.amostra, est.pessoas, est.i_juiz = (hsv, lms_juiz), pessoas, i_juiz
            est.status = "reproduzindo" if fonte.tipo == "video" else "ao vivo"
            pontos = dict(placar.pontos)
            est.cond.notify_all()
        if ev:
            print(f"[{t:7.2f}s] {ev[1]} +{ev[2]} {nomes[ev[0]]}  ->  {pontos}", flush=True)

    print("final:", {est.cfg["nomes"][k]: v for k, v in placar.pontos.items()}, flush=True)
    if gravador:
        gravador.release()


# ---------------------------------------------------------------- painel web

MAX_QUADRO = 10 << 20
MAX_VIDEO = 4 << 30


def servir(est, porta):
    pagina = (BASE / "painel.html").read_bytes()

    class Painel(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def responder(self, corpo=b"", tipo="application/json", codigo=200):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corpo)

        def erro(self, msg, codigo=400):
            self.responder(json.dumps({"erro": msg}).encode(), codigo=codigo)

        def corpo(self, limite):
            n = int(self.headers.get("Content-Length") or 0)
            if n > limite:
                raise ValueError("arquivo grande demais")
            return self.rfile.read(n)

        def do_GET(self):
            rota = urlparse(self.path).path
            if rota == "/":
                return self.responder(pagina, "text/html; charset=utf-8")
            if rota == "/estado":
                return self.responder(est.json().encode())
            if rota == "/videos":
                return self.responder(json.dumps(listar_videos(est.pasta_videos)).encode())
            if rota != "/video":
                return self.send_error(404)
            # MJPEG: o navegador mostra num <img> comum, um JPEG atrás do outro
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=quadro")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            visto = -1
            try:
                self.wfile.write(b"--quadro\r\n")
                while True:
                    with est.cond:
                        est.cond.wait_for(lambda: est.n != visto, timeout=5)
                        jpg, visto = est.jpg, est.n
                    # separador logo depois do JPEG: o Chrome só desenha um quadro quando vê o próximo separador
                    if jpg:
                        self.wfile.write(b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n--quadro\r\n" % (len(jpg), jpg))
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            url = urlparse(self.path)
            try:
                if url.path == "/quadro":
                    jpg = self.corpo(MAX_QUADRO)
                    fonte = est.fonte
                    if not isinstance(fonte, FonteNavegador):
                        return self.erro("a fonte atual não é a câmera do PC", 409)
                    return self.responder(codigo=204) if fonte.receber(jpg) else self.erro("quadro inválido")
                if url.path == "/enviar":
                    return self.enviar(nome_video(parse_qs(url.query).get("nome", [""])[0]))
                dados = json.loads(self.corpo(1 << 16) or b"{}")
                if url.path == "/fonte":
                    est.trocar_fonte(abrir_fonte(dados, est.pasta_videos))
                elif url.path == "/config":
                    est.configurar(dados)
                elif url.path == "/juiz":
                    if dados.get("acao") == "soltar":
                        with est.cond:
                            est.juiz.soltar()
                    else:
                        est.travar_juiz(dados.get("x"), dados.get("y"))
                elif url.path == "/calibrar":
                    return self.responder(json.dumps(est.calibrar()).encode())
                elif url.path in ("/desfazer", "/zerar", "/marcar", "/cronometro"):
                    with est.cond:
                        self.placar_ou_relogio(url.path, dados)
                else:
                    return self.send_error(404)
            except (ValueError, TypeError, AttributeError) as e:
                return self.erro(str(e))
            self.responder(codigo=204)

        def placar_ou_relogio(self, rota, dados):
            p, r = est.placar, est.relogio
            if rota == "/desfazer":
                p.desfazer()
            elif rota == "/zerar":
                p.zerar()
                r.reiniciar()
            elif rota == "/marcar":
                n = int(dados.get("n", 1))
                if not -10 <= n <= 10:
                    raise ValueError("valor inválido")
                p.marcar(dados.get("quem"), dados.get("tipo", "pontos"), n)
            elif dados.get("acao") == "alternar":
                r.alternar()
            elif dados.get("acao") == "reiniciar":
                duracao = int(dados.get("duracao") or 0)
                r.reiniciar(duracao if 30 <= duracao <= 3600 else None)
            else:
                raise ValueError("ação inválida")

        def enviar(self, nome):
            """Upload do vídeo direto pra pasta de vídeos, em pedaços, sem guardar tudo na memória."""
            falta = int(self.headers.get("Content-Length") or 0)
            if not 0 < falta <= MAX_VIDEO:
                raise ValueError("arquivo vazio ou grande demais")
            est.pasta_videos.mkdir(parents=True, exist_ok=True)
            tmp = est.pasta_videos / f".{nome}.parcial"
            try:
                with tmp.open("wb") as f:
                    while falta:
                        bloco = self.rfile.read(min(falta, 1 << 20))
                        if not bloco:
                            raise ValueError("envio interrompido")
                        f.write(bloco)
                        falta -= len(bloco)
                tmp.replace(est.pasta_videos / nome)
            finally:
                tmp.unlink(missing_ok=True)
            self.responder(json.dumps({"nome": nome}).encode())

    ThreadingHTTPServer.daemon_threads = True
    print(f"painel em http://localhost:{porta}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", porta), Painel).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", help="fonte inicial: índice da câmera, arquivo de vídeo ou URL (rtsp/http). "
                                    "Sem isso, escolha no painel.")
    ap.add_argument("--videos", default="videos", help="pasta dos vídeos que aparecem no painel (e recebe os enviados)")
    ap.add_argument("--modo", default="mao", choices=("mao", "faixa"),
                    help="como saber o atleta: mao = direita do juiz é o da faixa verde-amarela; faixa = cor no pulso")
    ap.add_argument("--inverter", action="store_true", help="troca esquerda/direita (câmera espelhada)")
    ap.add_argument("--cor", default="verde", choices=CORES, help="cor da faixa no pulso (modo faixa)")
    ap.add_argument("--nomes", default="Verde,Branco", help="nome do atleta da faixa, nome do outro")
    ap.add_argument("--min-frac", type=float, default=0.15, help="fração mínima de pixels da cor no pulso")
    ap.add_argument("--segurar", type=float, default=0.5, help="segundos segurando o gesto pra pontuar")
    ap.add_argument("--soltar", type=float, default=0.8, help="segundos sem gesto pra liberar o próximo ponto")
    ap.add_argument("--fps", type=float, default=15, help="máximo de quadros analisados por segundo, 5 a 60 (menos = mais leve)")
    ap.add_argument("--saida", help="grava o vídeo anotado (.mp4)")
    ap.add_argument("--porta", type=int, default=8000, help="porta do painel web; 0 = sem painel, processa e sai")
    a = ap.parse_args()

    nomes = a.nomes.split(",")
    if len(nomes) != 2:
        ap.error("--nomes precisa de dois nomes separados por vírgula, ex: Verde,Branco")
    if not a.porta and not a.fonte:
        ap.error("sem painel (--porta 0) precisa de --fonte")
    cfg = {"modo": a.modo, "inverter": a.inverter, "cor": a.cor, "nomes": dict(zip(("cor", "outro"), nomes)),
           "min_frac": a.min_frac, "fps": min(max(a.fps, 5), 60)}
    est = Estado(Placar(a.segurar, a.soltar), cfg, a.videos)

    if a.fonte:
        pedido, pasta = fonte_da_linha(a.fonte, a.videos)
        try:
            est.trocar_fonte(abrir_fonte(pedido, pasta, tempo_real=bool(a.porta)))
        except ValueError as e:
            if not a.porta:
                raise SystemExit(str(e))
            print(f"aviso: {e} — escolha outra fonte no painel", flush=True)
    if not a.porta:
        return processar(est, a.saida, painel=False)
    threading.Thread(target=processar, args=(est, a.saida), daemon=True).start()
    if getattr(sys, "frozen", False):  # executável: abre o painel sozinho
        threading.Timer(1.5, webbrowser.open, [f"http://localhost:{a.porta}"]).start()
    servir(est, a.porta)


if __name__ == "__main__":
    main()
