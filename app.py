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

    
import os
import subprocess
import tempfile
import gradio as gr
import imageio_ffmpeg
import numpy as np

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def analyser_video(video):
    if video is None:
        raise gr.Error("Ajoute une vidéo.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dossier = tempfile.mkdtemp()

    # Détection des changements de scène
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27))

    try:
        source = open_video(video)
        scene_manager.detect_scenes(source)
        scenes = scene_manager.get_scene_list()
    except Exception as e:
        raise gr.Error(f"Erreur d'analyse vidéo : {e}")

    # Récupérer la durée
    info = subprocess.run(
        [ffmpeg, "-i", video, "-f", "null", "-"],
        capture_output=True,
        text=True
    )

    import re
    match = re.search(r"Duration: (\d+):(\d+):(\d+)", info.stderr)

    if not match:
        raise gr.Error("Impossible de lire la durée.")

    h, m, s = map(int, match.groups())
    duree = h * 3600 + m * 60 + s

    # Créer des passages candidats autour des changements de scène
    candidats = []

    for debut_scene, fin_scene in scenes:
        debut = debut_scene.get_seconds()
        if debut < duree:
            candidats.append(debut)

    # Ajouter des passages régulièrement espacés
    for debut in range(0, duree, 30):
        candidats.append(float(debut))

    # Trier et supprimer les doublons proches
    candidats = sorted(set(candidats))

    # Sélectionner les passages de 30 secondes
    selection = []

if duree <= 30:
    selection.append(0.0)
else:
    for debut in candidats:
        if debut + 30 <= duree:
            if all(abs(debut - ancien) >= 20 for ancien in selection):
                selection.append(debut)

selection = selection[:10]

clips = []

    for index, debut in enumerate(selection):
        sortie = os.path.join(
            dossier,
            f"clip_{index + 1}.mp4"
        )

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
        ], capture_output=True, timeout=120)

        if os.path.exists(sortie) and os.path.getsize(sortie) > 0:
            clips.append((sortie, f"Clip {index + 1}"))

    if not clips:
        raise gr.Error("Aucun clip n'a pu être créé.")

    return clips


with gr.Blocks(title="ClipIA") as demo:
    gr.Markdown("# 🎬 ClipIA")
    gr.Markdown("Détection automatique des scènes intéressantes.")

    video = gr.Video(label="Choisis ta vidéo")

    bouton = gr.Button("Trouver les meilleurs moments")

    resultat = gr.Gallery(
        label="🎬 Tes meilleurs clips",
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
