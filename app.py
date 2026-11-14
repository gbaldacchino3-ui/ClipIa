

        
import os
import cv2
import subprocess
import tempfile
import gradio as gr
import imageio_ffmpeg


def analyser_video(video, nombre_clips):
    if video is None:
        raise gr.Error("Ajoute d'abord une vidéo.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dossier = tempfile.mkdtemp()

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise gr.Error("Impossible de lire la vidéo.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0 or total <= 0:
        capture.release()
        raise gr.Error("Vidéo invalide.")

    duree = total / fps
    nombre_clips = int(nombre_clips)

    longueur = min(30, duree)

    if duree < longueur:
        longueur = duree

    intervalles = []

    precedente = None
    scores = []

    compteur = 0

    while True:
        ok, frame = capture.read()

        if not ok:
            break

        if compteur % max(1, int(fps * 2)) == 0:
            petite = cv2.resize(frame, (160, 90))
            gris = cv2.cvtColor(
                petite, cv2.COLOR_BGR2GRAY
            )

            score = 0

            if precedente is not None:
                score = cv2.absdiff(
                    gris, precedente
                ).mean()

            scores.append((compteur / fps, score))
            precedente = gris

        compteur += 1

    capture.release()

    scores.sort(key=lambda x: x[1], reverse=True)

    for temps, score in scores:
        debut = max(0, min(
            temps - longueur / 2,
            duree - longueur
        ))

        if all(
            abs(debut - ancien) >= longueur
            for ancien in intervalles
        ):
            intervalles.append(debut)

        if len(intervalles) >= nombre_clips:
            break

    if not intervalles:
        intervalles = [
            i * max(0, duree - longueur) /
            max(1, nombre_clips - 1)
            for i in range(nombre_clips)
        ]

    clips = []

    for i, debut in enumerate(intervalles):
        sortie = os.path.join(
            dossier, f"clip_{i+1}.mp4"
        )

        subprocess.run(
            [
                ffmpeg, "-y",
                "-ss", str(debut),
                "-i", video,
                "-t", str(longueur),
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-c:a", "aac",
                "-movflags", "+faststart",
                sortie
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=True
        )

        clips.append(sortie)

    return clips


with gr.Blocks(title="ClipIA") as demo:

    gr.Markdown("# 🎬 ClipIA")
    gr.Markdown(
        "Détecte automatiquement les passages "
        "dynamiques de tes vidéos."
    )

    video = gr.Video(
        label="Choisis ta vidéo"
    )

    nombre = gr.Slider(
        minimum=1,
        maximum=5,
        value=3,
        step=1,
        label="Nombre de clips"
    )

    bouton = gr.Button(
        "🚀 Créer mes clips"
    )

    resultat = gr.Gallery(
        label="Tes clips générés"
    )

    bouton.click(
        analyser_video,
        inputs=[video, nombre],
        outputs=resultat
    )


demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(
        os.environ.get("PORT", 7860)
    )
    )
    
