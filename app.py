import os
import os
import re
import subprocess
import tempfile
import gradio as gr
import imageio_ffmpeg

def analyser_video(video):
    if video is None:
        raise gr.Error("Ajoute d'abord une vidéo.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    info = subprocess.run(
        [ffmpeg, "-i", video, "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=120
    )

    match = re.search(
        r"Duration: (\d+):(\d+):(\d+)",
        info.stderr
    )

    if not match:
        raise gr.Error("Impossible de lire la vidéo.")

    heures, minutes, secondes = map(int, match.groups())
    duree = heures * 3600 + minutes * 60 + secondes

    dossier = tempfile.mkdtemp()
    clips = []

    for debut in range(0, duree, 30):
        sortie = os.path.join(
            dossier, f"clip_{debut // 30 + 1}.mp4"
        )

        try:
            subprocess.run([
                ffmpeg, "-y",
                "-ss", str(debut),
                "-i", video,
                "-t", "30",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-movflags", "+faststart",
                sortie
            ], capture_output=True, timeout=120, check=True)

            if os.path.exists(sortie) and os.path.getsize(sortie) > 0:
                clips.append((
                    sortie,
                    f"Clip {len(clips) + 1}"
                ))

        except subprocess.TimeoutExpired:
            raise gr.Error("Le traitement prend trop de temps.")

        except subprocess.CalledProcessError:
            raise gr.Error("Erreur pendant la création d'un clip.")

    if not clips:
        raise gr.Error("Aucun clip n'a pu être créé.")

    return clips


with gr.Blocks(title="ClipIA") as demo:
    gr.Markdown("# 🎬 ClipIA")
    gr.Markdown(
        "Transforme tes vidéos en clips de 30 secondes."
    )

    video = gr.Video(label="Choisis ta vidéo")

    bouton = gr.Button("Créer mes clips")

    resultat = gr.Gallery(
        label="🎬 Tes clips",
        columns=2,
        allow_preview=True
    )

    bouton.click(
        fn=analyser_video,
        inputs=video,
        outputs=resultat
    )

demo.queue()

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
    )
