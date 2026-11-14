
import os
import re
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
        # Lire uniquement les métadonnées
        resultat = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", video],
            capture_output=True,
            text=True,
            timeout=15
        )

        match = re.search(
            r"Duration: (\d+):(\d+):(\d+\.\d+)",
            resultat.stderr
        )

        if not match:
            raise gr.Error("Impossible de lire la durée de la vidéo.")

        h, m, s = match.groups()
        duree = int(h) * 3600 + int(m) * 60 + float(s)

        nombre_clips = int(nombre_clips)
        longueur = min(30, duree / nombre_clips)

        clips = []

        for i in range(nombre_clips):
            debut = i * duree / nombre_clips
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
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    "-movflags", "+faststart",
                    sortie
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True
            )

            if not os.path.exists(sortie) or os.path.getsize(sortie) == 0:
                raise gr.Error("Un clip n'a pas été créé.")

            clips.append(sortie)

        return clips

    except subprocess.TimeoutExpired:
        raise gr.Error("Le traitement dépasse le délai autorisé.")
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Erreur FFmpeg : {e.stderr[-1000:]}")
    except Exception as e:
        raise gr.Error(f"Erreur : {str(e)}")

with gr.Blocks(title="ClipIA") as demo:
    gr.Markdown("# 🎬 ClipIA")
    gr.Markdown("Transforme tes vidéos en clips.")

    video = gr.Video(label="Choisis ta vidéo")

    nombre = gr.Slider(
        minimum=1,
        maximum=5,
        value=1,
        step=1,
        label="Nombre de clips"
    )

    bouton = gr.Button("🚀 Créer mes clips")
    resultat = gr.Gallery(label="Tes clips générés")

    bouton.click(
        analyser_video,
        inputs=[video, nombre],
        outputs=resultat
    )

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
                                                            )
 
