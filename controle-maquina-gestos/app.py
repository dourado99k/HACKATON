from datetime import datetime
from math import hypot
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)
app.secret_key = "controle-maquina-gestos"

gesto_atual = "nenhum"
maquina_ligada = False
historico = []
usuario_ativo = ""
easter_egg_ativo = False
easter_egg_contador = 0

MODELO_MAOS = Path(__file__).parent / "hand_landmarker.task"
PASTA_PRINTS = Path(__file__).parent / "prints"
PASTA_PRINTS.mkdir(exist_ok=True)

LARGURA_PROCESSAMENTO = 640
FPS_ALVO = 24
FRAMES_PARA_CONFIRMAR_GESTO = 2


def criar_detector_maos():
    base_options = python.BaseOptions(model_asset_path=str(MODELO_MAOS))
    opcoes_maos = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opcoes_maos)


def distancia(ponto_a, ponto_b):
    return hypot(ponto_a.x - ponto_b.x, ponto_a.y - ponto_b.y)


def dedo_estendido(landmarks, ponta, articulacao):
    pulso = landmarks[0]
    return distancia(landmarks[ponta], pulso) > distancia(landmarks[articulacao], pulso) * 1.05


def eh_joinha(landmarks):
    polegar_estendido = dedo_estendido(landmarks, 4, 3)
    indicador_dobrado = not dedo_estendido(landmarks, 8, 6)
    medio_dobrado = not dedo_estendido(landmarks, 12, 10)
    anelar_dobrado = not dedo_estendido(landmarks, 16, 14)
    mindinho_dobrado = not dedo_estendido(landmarks, 20, 18)

    return polegar_estendido and indicador_dobrado and medio_dobrado and anelar_dobrado and mindinho_dobrado


def eh_hangloose(landmarks):
    polegar_estendido = dedo_estendido(landmarks, 4, 3)
    mindinho_estendido = dedo_estendido(landmarks, 20, 18)
    indicador_dobrado = not dedo_estendido(landmarks, 8, 6)
    medio_dobrado = not dedo_estendido(landmarks, 12, 10)
    anelar_dobrado = not dedo_estendido(landmarks, 16, 14)

    return (
        polegar_estendido
        and mindinho_estendido
        and indicador_dobrado
        and medio_dobrado
        and anelar_dobrado
    )


def eh_dedo_meio(landmarks):
    medio_para_cima = landmarks[12].y < landmarks[10].y and landmarks[10].y < landmarks[9].y
    indicador_dobrado = landmarks[8].y > landmarks[6].y
    anelar_dobrado = landmarks[16].y > landmarks[14].y
    mindinho_dobrado = landmarks[20].y > landmarks[18].y

    medio_mais_alto = (
        landmarks[12].y < landmarks[8].y
        and landmarks[12].y < landmarks[16].y
        and landmarks[12].y < landmarks[20].y
    )

    medio_estendido = dedo_estendido(landmarks, 12, 10)
    outros_nao_estendidos = (
        not dedo_estendido(landmarks, 8, 6)
        and not dedo_estendido(landmarks, 16, 14)
        and not dedo_estendido(landmarks, 20, 18)
    )

    criterio_y = medio_para_cima and indicador_dobrado and anelar_dobrado and mindinho_dobrado
    criterio_relativo = medio_mais_alto and indicador_dobrado and anelar_dobrado
    criterio_distancia = medio_estendido and outros_nao_estendidos

    return criterio_y or criterio_relativo or criterio_distancia


def identificar_gesto(mao):
    if eh_dedo_meio(mao):
        return "dedo do meio"
    if eh_joinha(mao):
        return "joinha"
    if eh_hangloose(mao):
        return "hang loose"
    return "nenhum"


def desenhar_pontos(frame, landmarks):
    altura, largura, _ = frame.shape

    for ponto in landmarks:
        x = int(ponto.x * largura)
        y = int(ponto.y * altura)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def redimensionar_frame(frame):
    altura, largura = frame.shape[:2]

    if largura <= LARGURA_PROCESSAMENTO:
        return frame

    nova_largura = LARGURA_PROCESSAMENTO
    nova_altura = int(altura * (nova_largura / largura))
    return cv2.resize(frame, (nova_largura, nova_altura))


def salvar_print_webcam(frame, gesto, estado_maquina):
    momento = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    gesto_arquivo = gesto.replace(" ", "_")
    nome_arquivo = f"{estado_maquina}_{gesto_arquivo}_{momento}.jpg"
    caminho = PASTA_PRINTS / nome_arquivo
    cv2.imwrite(str(caminho), frame)
    return nome_arquivo


def registrar_acao(gesto, estado_maquina, print_arquivo):
    global historico

    historico.insert(
        0,
        {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "usuario": usuario_ativo or "Desconhecido",
            "horario": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "gesto": gesto,
            "estado_maquina": estado_maquina,
            "print": print_arquivo,
        },
    )


def caminho_print_seguro(nome_arquivo):
    caminho = (PASTA_PRINTS / Path(nome_arquivo).name).resolve()

    if not str(caminho).startswith(str(PASTA_PRINTS.resolve())):
        return None

    return caminho


def apagar_arquivo_print(nome_arquivo):
    caminho = caminho_print_seguro(nome_arquivo)

    if caminho is None or not caminho.exists():
        return False

    caminho.unlink()
    return True


def processar_easter_egg(detectado):
    global easter_egg_ativo, easter_egg_contador

    if detectado and not easter_egg_ativo:
        easter_egg_contador += 1
        easter_egg_ativo = True
    elif not detectado:
        easter_egg_ativo = False


def processar_gesto(novo_gesto, frame):
    global gesto_atual, maquina_ligada

    gesto_atual = novo_gesto

    if novo_gesto == "joinha" and not maquina_ligada:
        maquina_ligada = True
        print_arquivo = salvar_print_webcam(frame, novo_gesto, "ligada")
        registrar_acao("joinha", "ligada", print_arquivo)
    elif novo_gesto == "hang loose" and maquina_ligada:
        maquina_ligada = False
        print_arquivo = salvar_print_webcam(frame, novo_gesto, "desligada")
        registrar_acao("hang loose", "desligada", print_arquivo)


@app.route("/")
def home():
    if "usuario" in session:
        return redirect(url_for("painel"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    global usuario_ativo

    if request.method == "POST":
        nome = request.form.get("nome", "").strip()
        if nome:
            session["usuario"] = nome
            usuario_ativo = nome
            return redirect(url_for("painel"))

    return render_template("login.html")


@app.route("/painel")
def painel():
    if "usuario" not in session:
        return redirect(url_for("login"))

    return render_template("painel.html", usuario=session["usuario"])


@app.route("/logout")
def logout():
    global usuario_ativo

    session.clear()
    usuario_ativo = ""
    return redirect(url_for("login"))


def abrir_camera():
    backends = [
        (0, cv2.CAP_AVFOUNDATION),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_ANY),
    ]

    for indice, backend in backends:
        camera = cv2.VideoCapture(indice, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, FPS_ALVO)
            return camera
        camera.release()

    return None


def gerar_frames():
    global gesto_atual, maquina_ligada

    camera = abrir_camera()

    if camera is None:
        print("Erro: nao foi possivel abrir a camera.")
        return

    detector = criar_detector_maos()
    intervalo_frame = 1 / FPS_ALVO
    gesto_candidato = "nenhum"
    contagem_gesto = 0

    try:
        while True:
            inicio_frame = time.perf_counter()

            sucesso, frame = camera.read()

            if not sucesso:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            frame_processado = redimensionar_frame(frame)
            imagem_rgb = np.ascontiguousarray(cv2.cvtColor(frame_processado, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)

            resultado = detector.detect(mp_image)

            gesto_detectado = "nenhum"
            mao_detectada = False
            dedo_meio_detectado = False

            if resultado.hand_landmarks:
                mao_detectada = True
                for mao in resultado.hand_landmarks:
                    desenhar_pontos(frame_processado, mao)
                    gesto_mao = identificar_gesto(mao)

                    if gesto_mao == "dedo do meio":
                        dedo_meio_detectado = True
                        gesto_detectado = "nenhum"
                        cv2.putText(
                            frame_processado,
                            "DEDO DO MEIO!",
                            (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            4,
                        )
                    elif gesto_mao != "nenhum":
                        gesto_detectado = gesto_mao
                        texto = "JOINHA DETECTADO!" if gesto_mao == "joinha" else "HANG LOOSE DETECTADO!"
                        cv2.putText(
                            frame_processado,
                            texto,
                            (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            3,
                        )

            processar_easter_egg(dedo_meio_detectado)

            if mao_detectada and gesto_detectado == "nenhum" and not dedo_meio_detectado:
                cv2.putText(
                    frame_processado,
                    "Mao detectada - faca joinha ou hang loose",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
            elif not mao_detectada:
                cv2.putText(
                    frame_processado,
                    "Mostre a mao para a camera",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            if gesto_detectado == gesto_candidato and gesto_detectado != "nenhum":
                contagem_gesto += 1
            else:
                gesto_candidato = gesto_detectado
                contagem_gesto = 1 if gesto_detectado != "nenhum" else 0

            if contagem_gesto >= FRAMES_PARA_CONFIRMAR_GESTO and gesto_detectado != gesto_atual:
                processar_gesto(gesto_detectado, frame_processado.copy())
            elif gesto_detectado == "nenhum" and not dedo_meio_detectado:
                gesto_atual = "nenhum"

            status = "LIGADA" if maquina_ligada else "DESLIGADA"
            cor_status = (0, 255, 0) if maquina_ligada else (0, 0, 255)
            cv2.putText(
                frame_processado,
                f"Maquina: {status}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                cor_status,
                2,
            )

            ret, buffer = cv2.imencode(".jpg", frame_processado, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

            tempo_gasto = time.perf_counter() - inicio_frame
            tempo_espera = intervalo_frame - tempo_gasto
            if tempo_espera > 0:
                time.sleep(tempo_espera)
    finally:
        detector.close()
        camera.release()


@app.route("/video_feed")
def video_feed():
    if "usuario" not in session:
        return redirect(url_for("login"))

    return Response(
        gerar_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    if "usuario" not in session:
        return jsonify({"erro": "nao autenticado"}), 401

    return jsonify(
        {
            "gesto": gesto_atual,
            "maquina_ligada": maquina_ligada,
            "estado_maquina": "ligada" if maquina_ligada else "desligada",
            "easter_egg_contador": easter_egg_contador,
            "easter_egg_ativo": easter_egg_ativo,
                "total_historico": len(historico),
        }
    )


@app.route("/historico")
def obter_historico():
    if "usuario" not in session:
        return jsonify({"erro": "nao autenticado"}), 401

    return jsonify({"historico": historico})


@app.route("/prints/<path:nome_arquivo>", methods=["GET", "DELETE"])
def manipular_print(nome_arquivo):
    global historico

    if "usuario" not in session:
        if request.method == "DELETE":
            return jsonify({"erro": "nao autenticado"}), 401
        return redirect(url_for("login"))

    if request.method == "DELETE":
        if not apagar_arquivo_print(nome_arquivo):
            return jsonify({"erro": "print nao encontrado"}), 404

        nome_limpo = Path(nome_arquivo).name
        historico = [item for item in historico if item.get("print") != nome_limpo]

        return jsonify({"ok": True, "historico": historico})

    caminho = caminho_print_seguro(nome_arquivo)

    if caminho is None or not caminho.exists():
        return jsonify({"erro": "print nao encontrado"}), 404

    return send_from_directory(PASTA_PRINTS, caminho.name)


@app.route("/prints", methods=["DELETE"])
def apagar_todos_prints():
    global historico

    if "usuario" not in session:
        return jsonify({"erro": "nao autenticado"}), 401

    for arquivo in PASTA_PRINTS.glob("*.jpg"):
        arquivo.unlink()

    historico = []

    return jsonify({"ok": True, "historico": historico})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
