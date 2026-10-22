# Placar JJ por gesto

O juiz faz os sinais da IBJJF com a mão e o placar marca sozinho:

| Sinal | Como | Marca |
|---|---|---|
| ✌️ 2, 3 ou 4 dedos | pulso **acima dos ombros** do juiz (linha amarela) | 2 / 3 / 4 pontos |
| ✋ mão espalmada, palma pra baixo | braço **esticado na horizontal**, na altura do ombro | vantagem |
| ✊ punho fechado | braço levantado na altura do ombro | punição |

O sinal precisa ficar parado 0.5s. Duas mãos sinalizando ao mesmo tempo não marcam nada.

**Só as mãos do juiz contam.** Um modelo de pose (MediaPipe Pose *lite*) acha as pessoas na imagem; o juiz ganha caixa roxa
e as mãos dos outros aparecem como "ignorada". Sem travar, o juiz é a maior pessoa da imagem (a mais perto da câmera).
Com atletas/plateia por perto, **clique no juiz no vídeo** (ou *Travar juiz*): ele passa a ser seguido pela posição e pela
cor da roupa, e se sair e voltar é reencontrado. A altura dos ombros vem do corpo dele, então funciona perto ou longe da câmera.

**De quem é o sinal** (escolha em *Ajustes*):
- **Mão do juiz** (padrão): pela regra da IBJJF o juiz usa sempre a mesma mão pro mesmo atleta, onde quer que eles estejam
  no tatame. **Direita = atleta da faixa verde-amarela**, esquerda = o outro. Não precisa de faixa nenhuma.
  Se aparecer o lado trocado em *Mãos do juiz* (imagem espelhada), marque *Inverter esquerda/direita*.
- **Faixa no pulso**: a mão com a munhequeira colorida marca pro atleta da cor. Use *Calibrar faixa* com a faixa dentro do
  círculo: o painel aprende a cor e descarta tons de pele. Cores fortes (verde, azul) funcionam melhor que laranja/rosa.

## Painel (http://localhost:8000)

- **Fonte de vídeo** escolhida no próprio painel:
  - **Vídeo**: arquivos da pasta `videos/`, ou o botão *Enviar vídeo…* pra mandar um do PC.
  - **Câmera do PC**: a webcam é aberta pelo navegador e os quadros vão pro servidor, então funciona mesmo
    com o servidor no Docker/WSL. Só funciona abrindo o painel por `http://localhost` (regra do navegador) e com a aba aberta.
  - **Câmera IP**: link do celular/câmera (`http://IP:8080/video`, `rtsp://…`). Reconecta sozinho se o sinal cair.
- Placar com **vantagens e punições**, botões pra marcar à mão (+2 +3 +4 V P), histórico, desfazer (Ctrl+Z) e zerar.
- **Cronômetro** da luta (4–10 min, Espaço inicia/pausa, apita no fim).
- **Ajustes ao vivo**: nomes, modo mão/faixa, cor da faixa, sensibilidade da cor, tempos de segurar/soltar.

```bash
docker compose up --build                                     # abre com o vídeo de exemplo (videos/teste.mp4)
FONTE=http://192.168.0.50:8080/video docker compose up         # já abre na câmera do celular
FONTE=... COR=azul NOMES="Azul,Branco" docker compose up
docker compose run --rm --entrypoint python placar test_placar.py   # self-check
```

## Câmera do celular

No painel: aba **Câmera IP**, cole o link e *Conectar*.

1. Android: app **IP Webcam** → "Iniciar servidor"; ele mostra `http://IP:8080`. A fonte é `http://IP:8080/video`.
   iPhone: **DroidCam** (fonte `http://IP:4747/video`) ou qualquer app que dê um link RTSP/MJPEG.
2. Celular e PC na **mesma rede Wi-Fi**. Teste o link no navegador do PC antes.
3. No app, resolução **1280x720** (mais que isso só deixa lento) e foco/exposição travados se der.

Onde colocar:
- Num **tripé fixo** (nunca na mão), na lateral do tatame, de frente pro juiz, uns 3–5 m dele.
- Na **altura do peito ou um pouco acima**, deitado (paisagem), enquadrando o juiz da cintura até uns 50 cm acima da cabeça.
  Assim a mão levantada sempre fica dentro da imagem. Os ombros e o quadril dele visíveis ajudam o modelo de pose.
- A **faixa virada pra câmera**. Fundo atrás das mãos sem a cor da faixa (nada de banner verde se a faixa é verde).
- Juiz anda pelo tatame: se ele sair muito do quadro, abra mais o enquadramento antes de aproximar.

## Ajustes

Tudo isso dá pra mudar no painel (*Ajustes*); pela linha de comando são os valores iniciais.
`--min-frac` quanto da cor precisa aparecer no pulso (o painel mostra a % ao vivo; ponha uns 2/3 do valor que aparece com a faixa visível) ·
`--segurar` / `--soltar` segundos · `--fps` quadros analisados por segundo, 5 a 60 (padrão 15: o gesto dura 0.5s; em máquina fraca, 10 ainda funciona; também no painel e em `FPS=30 docker compose up`) · `--videos` pasta dos vídeos do painel · `--saida x.mp4` grava o vídeo anotado ·
`--modo mao|faixa` · `--inverter` · `--porta 0` sem painel, processa a `--fonte` e sai.
Pra mudar no compose: `docker compose run --rm --service-ports placar --fonte ... --modo faixa --cor azul`.
