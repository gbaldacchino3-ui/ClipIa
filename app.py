
import os
import cv2
import subprocess
import tempfile
import gradio as gr
import imageio_ffmpeg


def analyser_video(video, nombre_clips, format_video):
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

    intervalles = []
    scores = []
    precedente = None

    # Analyse rapide : une image toutes les 2 secondes
    temps = 0

    while temps < duree:
        capture.set(cv2.CAP_PROP_POS_MSEC, temps * 1000)
        ok, frame = capture.read()

        if not ok:
            temps += 2
            continue

        petite = cv2.resize(frame, (160, 90))
        gris = cv2.cvtColor(
            petite, cv2.COLOR_BGR2GRAY
        )

        score = 0

        if precedente is not None:
            score = cv2.absdiff(
                gris, precedente
            ).mean()

        scores.append((temps, score))
        precedente = gris

        temps += 2

    capture.release()

    scores.sort(key=lambda x: x[1], reverse=True)

    for temps, score in scores:
        debut = max(
            0,
            min(temps - longueur / 2, duree - longueur)
        )

        if all(
            abs(debut - ancien) >= longueur
            for ancien in intervalles
        ):
            intervalles.append(debut)

        if len(intervalles) >= nombre_clips:
            break

    # Compléter si nécessaire
    for i in range(nombre_clips):
        if len(intervalles) >= nombre_clips:
            break

        debut = i * max(0, duree - longueur) / max(
            1, nombre_clips - 1
        )

        if all(
            abs(debut - ancien) >= longueur
            for ancien in intervalles
        ):
            intervalles.append(debut)

    intervalles = sorted(intervalles[:nombre_clips])

    clips = []

    for i, debut in enumerate(intervalles):
        sortie = os.path.join(
            dossier, f"clip_{i+1}.mp4"
        )

        commande = [
            ffmpeg, "-y",
            "-ss", str(debut),
            "-i", video,
            "-t", str(longueur),
            "-map", "0:v:0",
            "-map", "0:a?",
        ]

        if format_video == "Vertical 9:16":
            filtre = (
                "[0:v]split=2[bg][fg];"
                "[bg]scale=720:1280:force_original_aspect_ratio=increase,"
                "crop=720:1280,boxblur=20:10[background];"
                "[fg]scale=720:1280:force_original_aspect_ratio=decrease,"
                "setsar=1[foreground];"
                "[background][foreground]"
                "overlay=(W-w)/2:(H-h)/2,format=yuv420p[out]"
            )

            commande.extend([
                "-filter_complex", filtre,
                "-map", "[out]"
            ])

            # Retirer le mapping vidéo original
            commande.remove("0:v:0")

        commande.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            sortie
        ])

        try:
            subprocess.run(
                commande,
                capture_output=True,
                text=True,
                timeout=180,
                check=True
            )
        except subprocess.TimeoutExpired:
            raise gr.Error("Le traitement est trop long.")
        except subprocess.CalledProcessError as e:
            raise gr.Error(
                f"Erreur FFmpeg : {e.stderr[-1000:]}"
            )

        if not os.path.exists(sortie) or os.path.getsize(sortie) == 0:
            raise gr.Error("Un clip n'a pas été créé.")

        clips.append(sortie)

    return clips


with gr.Blocks(title="ClipIA") as demo:

    gr.Markdown("# 🎬 ClipIA")
    gr.Markdown(
        "Transforme automatiquement tes vidéos en clips."
    )

    video = gr.Video(label="Choisis ta vidéo")

    nombre = gr.Slider(
        minimum=1,
        maximum=5,
        value=3,
        step=1,
        label="Nombre de clips"
    )

    format_video = gr.Radio(
        choices=["Original", "Vertical 9:16"],
        value="Original",
        label="Format des clips"
    )

    bouton = gr.Button("🚀 Créer mes clips")

    resultat = gr.Gallery(
        label="Tes clips générés"
    )

    bouton.click(
        analyser_video,
        inputs=[video, nombre, format_video],
        outputs=resultat
    )


demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
    )
        
