import os
import subprocess
import tempfile
import gradio as gr
import imageio_ffmpeg

def analyser_video(video, nombre_clips):
    if video is None:
        raise gr.Error("Ajoute d'abord une vidéo.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dossier = tempfile.mkdtemp()

    try:
        probe = subprocess.run(
            [ffmpeg, "-i", video, "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=120
        )

        import re
        texte = probe.stderr

        duree_match = re.search(
            r"Duration: (\d+):(\d+):(\d+\.\d+)",
            texte
        )

        if not duree_match:
            raise gr.Error("Impossible de lire la durée de la vidéo.")

        h, m, s = duree_match.groups()
        duree = int(h) * 3600 + int(m) * 60 + float(s)

        clips = []

        nombre_clips = min(int(nombre_clips), 5)
        longueur = min(30, duree / nombre_clips)

        for i in range(nombre_clips):
            debut = i * longueur
            sortie = os.path.join(dossier, f"clip_{i+1}.mp4")

            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-ss", str(debut),
                    "-i", video,
                    "-t", str(longueur),
                    "-vf", "scale=-2:720",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "28",
                    "-c:a", "aac",
                    "-b:a", "96k",
                    "-movflags", "+faststart",
                    sortie
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True
            )

            clips.append(sortie)

        return clips

    except subprocess.TimeoutExpired:
        raise gr.Error("Le traitement prend trop de temps. Essaie une vidéo plus courte.")
    except Exception as e:
        raise gr.Error(f"Erreur : {str(e)}")

with gr.Blocks(title="ClipIA") as demo:
    gr.Markdown("# 🎬 ClipIA")
    gr.Markdown("Transforme tes vidéos en clips automatiquement.")

    video = gr.Video(label="Choisis ta vidéo")
    nombre = gr.Slider(
        minimum=1,
        maximum=5,
        value=3,
        step=1,
        label="Nombre de clips"
    )

    bouton = gr.Button("🚀 Créer mes clips")
    resultat = gr.Gallery(label="Tes clips générés")

    bouton.click(
        fn=analyser_video,
        inputs=[video, nombre],
        outputs=resultat
    )

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
                       
