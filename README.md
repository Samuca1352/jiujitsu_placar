# Placar JJ por gesto

Juiz levanta a mão (pulso acima da linha amarela) com ✌️ 2, 3 ou 4 dedos e segura por 0.5s: pontua.
Mão com a faixa colorida no pulso pontua pro atleta da cor; a outra mão pontua pro adversário.
Painel web em **http://localhost:8000**: câmera com as mãos detectadas, placar, histórico, desfazer/zerar.

```bash
docker compose up --build                                     # vídeo de exemplo (videos/teste.mp4)
FONTE=http://192.168.0.50:8080/video docker compose up         # câmera do celular
FONTE=... COR=azul NOMES="Azul,Branco" docker compose up
docker compose run --rm --entrypoint python placar test_placar.py   # self-check
```

## Câmera do celular

1. Android: app **IP Webcam** → "Iniciar servidor"; ele mostra `http://IP:8080`. A fonte é `http://IP:8080/video`.
   iPhone: **DroidCam** (fonte `http://IP:4747/video`) ou qualquer app que dê um link RTSP/MJPEG.
2. Celular e PC na **mesma rede Wi-Fi**. Teste o link no navegador do PC antes.
3. No app, resolução **1280x720** (mais que isso só deixa lento) e foco/exposição travados se der.

Onde colocar:
- Num **tripé fixo** (nunca na mão), na lateral do tatame, de frente pro juiz, uns 3–5 m dele.
- Na **altura do peito ou um pouco acima**, deitado (paisagem), enquadrando o juiz da cintura até uns 50 cm acima da cabeça.
  Assim a mão levantada sempre fica dentro da imagem e a linha amarela (`--linha`) fica na altura dos ombros dele.
- A **faixa virada pra câmera**. Fundo atrás das mãos sem a cor da faixa (nada de banner verde se a faixa é verde).
- Juiz anda pelo tatame: se ele sair muito do quadro, abra mais o enquadramento antes de aproximar.

## Ajustes

`--linha` altura da mão levantada (0=topo, 1=base) · `--min-frac` quanto da cor precisa aparecer no pulso
(o painel mostra a % ao vivo; ponha uns 2/3 do valor que aparece com a faixa visível) · `--segurar` / `--soltar` segundos ·
`--saida x.mp4` grava o vídeo anotado · `--porta 0` sem painel, processa e sai.
Pra mudar no compose: `docker compose run --rm --service-ports placar --fonte ... --linha 0.4`.
