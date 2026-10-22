"""Placar de jiu-jitsu por gesto do juiz.

Juiz levanta a mão acima da linha com 2, 3 ou 4 dedos -> pontos.
Mão com a faixa colorida no pulso -> atleta "cor"; mão sem faixa -> o outro.
Painel web em http://localhost:8000: vídeo com as caixas, placar, desfazer/zerar.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

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


def contar_dedos(lm):
    """Dedos esticados (sem polegar): ponta mais longe do pulso que a PIP. Não depende da rotação da mão."""
    p = np.array([(l.x, l.y) for l in lm])
    d = lambda i: np.linalg.norm(p[i] - p[0])
    return int(sum(d(tip) > d(pip) * 1.1 for tip, pip in DEDOS))


def frac_faixa(hsv, lm, faixas):
    """Fração da cor num círculo logo abaixo do pulso, no sentido do antebraço. Retorna (frac, centro, raio)."""
    h, w = hsv.shape[:2]
    pulso = np.array([lm[0].x * w, lm[0].y * h])
    meio = np.array([lm[9].x * w, lm[9].y * h])
    cx, cy = (pulso + (pulso - meio) * 0.4).astype(int)
    raio = max(int(np.linalg.norm(meio - pulso) * 0.5), 6)
    x0, y0, x1, y1 = max(cx - raio, 0), max(cy - raio, 0), min(cx + raio + 1, w), min(cy + raio + 1, h)
    if x0 >= x1 or y0 >= y1:
        return 0.0, (int(cx), int(cy)), raio
    rec = hsv[y0:y1, x0:x1]
    roi = np.zeros(rec.shape[:2], np.uint8)
    cv2.circle(roi, (int(cx - x0), int(cy - y0)), raio, 255, -1)
    cor = np.zeros_like(roi)
    for lo, hi in faixas:
        cor |= cv2.inRange(rec, np.array(lo), np.array(hi))
    area = cv2.countNonZero(roi)
    return (cv2.countNonZero(cor & roi) / area if area else 0.0), (int(cx), int(cy)), raio


def quem(frac, min_frac):
    """Faixa clara -> "cor"; quase nada da cor -> "outro"; meio-termo (faixa meio escondida) -> None."""
    return "cor" if frac >= min_frac else "outro" if frac < min_frac / 3 else None


def escolher(maos):
    """maos = [(levantada, dedos, quem)]. Retorna o gesto, None (ninguém sinalizando) ou PULAR (ambíguo)."""
    g = [(q, d) for lev, d, q in maos if lev and d in PONTOS_VALIDOS]
    if len(g) > 1 or any(q is None for q, _ in g):
        return PULAR
    return g[0] if g else None


class Placar:
    """Gesto precisa ficar parado `segurar` segundos; depois só pontua de novo quando a mão abaixa por `soltar`."""

    def __init__(self, segurar=0.5, soltar=0.8):
        self.segurar, self.soltar = segurar, soltar
        self.zerar()

    def zerar(self):
        self.pontos = {"cor": 0, "outro": 0}
        self.historico = []  # (quem, pontos, segundos)
        self.candidato, self.desde, self.travado, self.vazio_desde = None, 0.0, False, None

    def update(self, gesto, t):
        """gesto = ("cor"|"outro", pontos) ou None. Retorna o gesto quando pontua."""
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
            self.pontos[gesto[0]] += gesto[1]
            self.historico.append((*gesto, round(t, 1)))
            return gesto
        return None

    def desfazer(self):
        if self.historico:
            q, n, _ = self.historico.pop()
            self.pontos[q] -= n


class UltimoFrame:
    """Lê a câmera ao vivo numa thread e guarda só o frame mais novo, senão o buffer do stream vira atraso."""

    def __init__(self, cap):
        self.cap, self.ok, self.frame, self.novo = cap, True, None, threading.Event()
        threading.Thread(target=self._ler, daemon=True).start()

    def _ler(self):
        while self.ok:
            self.ok, self.frame = self.cap.read()
            self.novo.set()

    def read(self):
        self.novo.wait()
        self.novo.clear()
        return self.ok, self.frame


class Estado:
    """O que a thread de vídeo publica e o painel lê. `cond` protege tudo, inclusive o placar."""

    def __init__(self, placar, nomes, cor):
        self.cond, self.placar, self.jpg, self.n = threading.Condition(), placar, None, 0
        self.info = {"nomes": nomes, "cor": cor, "css": CSS[cor], "maos": [], "fps": 0.0, "status": "abrindo"}

    def json(self):
        with self.cond:
            return json.dumps({**self.info, "pontos": self.placar.pontos, "historico": self.placar.historico})


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


def processar(a, cap, eh_arquivo, est):
    nomes, placar = est.info["nomes"], est.placar
    cor_bgr = tuple(int(CSS[a.cor][i:i + 2], 16) for i in (5, 3, 1))
    det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"), running_mode=RunningMode.VIDEO, num_hands=2))
    fonte = cap if eh_arquivo else UltimoFrame(cap)
    gravador, ultimo_t, inicio, fps, antes = None, -1, time.monotonic(), 0.0, time.monotonic()
    est.info["status"] = "ao vivo"

    while True:
        ok, frame = fonte.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        # timestamp do vídeo em arquivo; relógio desde o início em câmera ao vivo
        t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) if eh_arquivo else int((time.monotonic() - inicio) * 1000)
        t_ms = ultimo_t = max(t_ms, ultimo_t + 1)
        if eh_arquivo and a.porta:  # no painel, arquivo toca em tempo real em vez de correr
            time.sleep(max(0.0, inicio + t_ms / 1000 - time.monotonic()))
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), t_ms)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        maos, info_maos = [], []
        for lm in res.hand_landmarks:
            dedos, levantada = contar_dedos(lm), lm[0].y < a.linha
            frac, centro, raio = frac_faixa(hsv, lm, CORES[a.cor])
            q = quem(frac, a.min_frac)
            maos.append((levantada, dedos, q))
            info_maos.append({"dedos": dedos, "faixa": round(frac, 2), "levantada": levantada, "quem": nomes.get(q)})
            pontua = levantada and dedos in PONTOS_VALIDOS
            caixa = cor_bgr if q == "cor" else (235, 235, 235) if q == "outro" else (0, 165, 255)
            rotulo = f"+{dedos} {nomes[q]}" if pontua and q else f"{dedos} dedos" + ("" if q else " faixa?")
            desenhar_mao(frame, lm, caixa if levantada else (140, 140, 140), f"{rotulo} {frac:.0%}", pontua)
            cv2.circle(frame, centro, raio, caixa, 1)

        gesto = escolher(maos)
        with est.cond:
            ev = None if gesto == PULAR else placar.update(gesto, t_ms / 1000)
            pontos = dict(placar.pontos)
        if ev:
            print(f"[{t_ms / 1000:7.2f}s] +{ev[1]} {nomes[ev[0]]}  ->  {pontos}", flush=True)

        cv2.line(frame, (0, int(a.linha * h)), (w, int(a.linha * h)), (0, 255, 255), 1)
        cv2.putText(frame, f"{nomes['cor']} {pontos['cor']} x {pontos['outro']} {nomes['outro']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if a.saida:
            gravador = gravador or cv2.VideoWriter(a.saida, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
            gravador.write(frame)
        agora = time.monotonic()
        fps, antes = 0.9 * fps + 0.1 / max(agora - antes, 1e-3), agora
        if a.porta:
            jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
            with est.cond:
                est.jpg, est.n = jpg, est.n + 1
                est.info.update(maos=info_maos, fps=round(fps, 1))
                est.cond.notify_all()

    est.info["status"] = "fim do vídeo" if eh_arquivo else "câmera caiu"
    print("final:", {nomes[k]: v for k, v in placar.pontos.items()}, flush=True)
    if gravador:
        gravador.release()


def servir(est, porta):
    pagina = Path(__file__).with_name("painel.html").read_bytes()

    class Painel(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def responder(self, corpo, tipo):
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corpo)

        def do_GET(self):
            if self.path == "/":
                return self.responder(pagina, "text/html; charset=utf-8")
            if self.path == "/estado":
                return self.responder(est.json().encode(), "application/json")
            if self.path != "/video":
                return self.send_error(404)
            # MJPEG: o navegador mostra num <img> comum, um JPEG atrás do outro
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=quadro")
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
            acao = {"/desfazer": est.placar.desfazer, "/zerar": est.placar.zerar}.get(self.path)
            if not acao:
                return self.send_error(404)
            with est.cond:
                acao()
            self.send_response(204)
            self.end_headers()

    ThreadingHTTPServer.daemon_threads = True
    print(f"painel em http://localhost:{porta}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", porta), Painel).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", default="0", help="índice da câmera, arquivo de vídeo ou URL (rtsp/http)")
    ap.add_argument("--cor", default="verde", choices=CORES)
    ap.add_argument("--nomes", default="Verde,Branco", help="nome do atleta da faixa, nome do outro")
    ap.add_argument("--linha", type=float, default=0.45, help="pulso acima disso (0=topo, 1=base) = mão levantada")
    ap.add_argument("--min-frac", type=float, default=0.15, help="fração mínima de pixels da cor no pulso")
    ap.add_argument("--segurar", type=float, default=0.5, help="segundos segurando o gesto pra pontuar")
    ap.add_argument("--soltar", type=float, default=0.8, help="segundos sem gesto pra liberar o próximo ponto")
    ap.add_argument("--saida", help="grava o vídeo anotado (.mp4)")
    ap.add_argument("--porta", type=int, default=8000, help="porta do painel web; 0 = sem painel, processa e sai")
    a = ap.parse_args()

    nomes = a.nomes.split(",")
    if len(nomes) != 2:
        ap.error("--nomes precisa de dois nomes separados por vírgula, ex: Verde,Branco")
    eh_arquivo = not a.fonte.isdigit() and "://" not in a.fonte
    cap = cv2.VideoCapture(int(a.fonte) if a.fonte.isdigit() else a.fonte)
    if not cap.isOpened():
        raise SystemExit(f"não abriu a fonte: {a.fonte}")
    est = Estado(Placar(a.segurar, a.soltar), dict(zip(("cor", "outro"), nomes)), a.cor)
    if not a.porta:
        return processar(a, cap, eh_arquivo, est)
    threading.Thread(target=processar, args=(a, cap, eh_arquivo, est), daemon=True).start()
    servir(est, a.porta)


if __name__ == "__main__":
    main()
