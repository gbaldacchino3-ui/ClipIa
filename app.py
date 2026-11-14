import os
import cv2
import re
import subprocess
import tempfile
import shutil
import math
import gradio as gr
import imageio_ffmpeg

from faster_whisper import WhisperModel


# ============================================================
# CONFIGURATION CLIPIA
# ============================================================

NOMBRE_MAX_CLIPS = 5
DUREE_MAX_CLIP = 45
INTERVALLE_ANALYSE = 3
MODELE_WHISPER = "tiny"

modele_whisper = None


# ============================================================
# CHARGEMENT DE WHISPER
# ============================================================

def charger_whisper():
    global modele_whisper

    if modele_whisper is None:
        modele_whisper = WhisperModel(
            MODELE_WHISPER,
            device="cpu",
            compute_type="int8"
        )

    return modele_whisper


# ============================================================
# VERIFICATION DE LA VIDEO
# ============================================================

def verifier_video(video):

    if video is None:
        raise gr.Error("Ajoute d'abord une vidéo.")

    if isinstance(video, dict):
        video = video.get("path") or video.get("name")

    if not isinstance(video, str):
        raise gr.Error("Le fichier vidéo n'a pas été reconnu.")

    if not os.path.isfile(video):
        raise gr.Error("Le fichier vidéo est introuvable.")

    if os.path.getsize(video) == 0:
        raise gr.Error("Le fichier vidéo est vide.")

    return video


# ============================================================
# INFORMATIONS VIDEO
# ============================================================

def obtenir_infos_video(video):

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        capture.release()
        raise gr.Error("Impossible d'ouvrir cette vidéo.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0 or total_frames <= 0:
        capture.release()
        raise gr.Error("Impossible de lire les informations de la vidéo.")

    duree = total_frames / fps

    capture.release()

    return fps, int(total_frames), duree


# ============================================================
# EXTRACTION AUDIO
# ============================================================

def extraire_audio(video, dossier):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    audio = os.path.join(dossier, "audio.wav")

    commande = [
        ffmpeg,
        "-y",
        "-i", video,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        audio
    ]

    resultat = subprocess.run(
        commande,
        capture_output=True,
        text=True,
        timeout=180
    )

    if resultat.returncode != 0 or not os.path.isfile(audio):
        return None

    if os.path.getsize(audio) == 0:
        return None

    return audio


# ============================================================
# ANALYSE AUDIO AVEC WHISPER
# ============================================================

def analyser_audio(audio, duree):

    if audio is None:
        return []

    try:
        modele = charger_whisper()

        segments, informations = modele.transcribe(
            audio,
            language="fr",
            beam_size=1,
            vad_filter=True
        )

        passages = []

        for segment in segments:

            texte = segment.text.strip()

            if not texte:
                continue

            debut = max(0, float(segment.start))
            fin = min(duree, float(segment.end))

            if fin <= debut:
                continue

            score = 0

            mots = texte.split()
            score += min(len(mots) * 1.5, 20)

            if "!" in texte:
                score += 8

            if "?" in texte:
                score += 5

            if texte.isupper() and len(texte) > 5:
                score += 5

            score += max(0, 10 - float(segment.no_speech_prob) * 10)

            passages.append({
                "debut": debut,
                "fin": fin,
                "score": score,
                "texte": texte
            })

        return passages

    except Exception as erreur:
        print("Erreur Whisper :", erreur)
        return []


# ============================================================
# ANALYSE VISUELLE
# ============================================================

def analyser_visuellement(video, fps, total_frames, duree):

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        return []

    resultats = []
    ancienne_image = None

    temps = 0

    while temps < duree:

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            temps * 1000
        )

        ok, image = capture.read()

        if not ok:
            temps += INTERVALLE_ANALYSE
            continue

        image = cv2.resize(image, (320, 180))

        gris = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        nettete = cv2.Laplacian(
            gris,
            cv2.CV_64F
        ).var()

        mouvement = 0

        if ancienne_image is not None:

            difference = cv2.absdiff(
                gris,
                ancienne_image
            )

            mouvement = float(difference.mean())

        score = min(nettete / 15, 20)
        score += min(mouvement * 1.5, 20)

        resultats.append({
            "temps": temps,
            "score": score
        })

        ancienne_image = gris.copy()

        temps += INTERVALLE_ANALYSE

    capture.release()

    return resultats


# ============================================================
# CREATION DES CANDIDATS
# ============================================================

def creer_candidats(duree, audio, visuel):

    candidats = []

    duree_clip = min(30, DUREE_MAX_CLIP, duree)

    if duree_clip <= 0:
        return []

    pas = max(2, duree_clip / 3)

    debut = 0

    while debut + duree_clip <= duree + 0.01:

        fin = min(debut + duree_clip, duree)

        score_audio = 0
        score_visuel = 0

        for passage in audio:

            milieu = (
                passage["debut"] + passage["fin"]
            ) / 2

            if debut <= milieu <= fin:
                score_audio += passage["score"]

        for image in visuel:

            if debut <= image["temps"] <= fin:
                score_visuel += image["score"]

        score_total = score_audio + score_visuel

        candidats.append({
            "debut": debut,
            "fin": fin,
            "score": score_total
        })

        debut += pas

    return candidats


# ============================================================
# SELECTION DES PASSAGES DISTINCTS
# ============================================================

def selectionner_passages(candidats, nombre):

    candidats = sorted(
        candidats,
        key=lambda x: x["score"],
        reverse=True
    )

    selection = []

    for candidat in candidats:

        if len(selection) >= nombre:
            break

        debut = candidat["debut"]
        fin = candidat["fin"]

        chevauchement = False

        for choisi in selection:

            autre_debut = choisi["debut"]
            autre_fin = choisi["fin"]

            if debut < autre_fin + 1 and fin > autre_debut - 1:
                chevauchement = True
                break

        if not chevauchement:
            selection.append(candidat)

    return sorted(
        selection,
        key=lambda x: x["debut"]
    )


# ============================================================
# CREATION D'UN CLIP
# ============================================================

def creer_clip(
    video,
    debut,
    fin,
    sortie,
    format_video
):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    duree = fin - debut

    if duree <= 0:
        raise RuntimeError("Durée de clip invalide.")

    commande = [
        ffmpeg,
        "-y",
        "-ss", str(debut),
        "-i", video,
        "-t", str(duree)
    ]

    if format_video == "Vertical 9:16":

        filtre = (
            "[0:v]split=2[arriere][avant];"
            "[arriere]scale=480:854:force_original_aspect_ratio=increase,"
            "crop=480:854,boxblur=20:10[fond];"
            "[avant]scale=480:854:force_original_aspect_ratio=decrease"
            "[image];"
            "[fond][image]overlay=(W-w)/2:(H-h)/2"
        )

        commande += [
            "-filter_complex", filtre,
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            sortie
        ]

    else:

        commande += [
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            sortie
        ]

    resultat = subprocess.run(
        commande,
        capture_output=True,
        text=True,
        timeout=240
    )

    if resultat.returncode != 0:

        message = resultat.stderr or "Erreur FFmpeg inconnue."

        raise RuntimeError(
            "Erreur FFmpeg : " + message[-1200:]
        )

    if not os.path.isfile(sortie):
        raise RuntimeError("Le clip n'a pas été créé.")

    if os.path.getsize(sortie) == 0:
        raise RuntimeError("Le clip créé est vide.")

    return sortie


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def analyser_video(
    video,
    nombre_clips,
    format_video,
    progress=gr.Progress()
):

    video = verifier_video(video)

    dossier = tempfile.mkdtemp(
        prefix="clipia_"
    )

    try:

        progress(0.05, desc="Lecture de la vidéo...")

        fps, total_frames, duree = obtenir_infos_video(video)

        if duree < 2:
            raise gr.Error("La vidéo est trop courte.")

        nombre_clips = int(nombre_clips)

        nombre_clips = max(
            1,
            min(nombre_clips, NOMBRE_MAX_CLIPS)
        )

        progress(0.12, desc="Extraction de l'audio...")

        audio_fichier = extraire_audio(
            video,
            dossier
        )

        progress(0.20, desc="Analyse de la voix avec Whisper...")

        audio = analyser_audio(
            audio_fichier,
            duree
        )

        progress(0.45, desc="Analyse des images...")

        visuel = analyser_visuellement(
            video,
            fps,
            total_frames,
            duree
        )

        progress(0.60, desc="Recherche des meilleurs passages...")

        candidats = creer_candidats(
            duree,
            audio,
            visuel
        )

        if not candidats:
            raise gr.Error("Aucun passage exploitable trouvé.")

        selection = selectionner_passages(
            candidats,
            nombre_clips
        )

        if not selection:
            raise gr.Error("Impossible de sélectionner des passages.")

        progress(0.70, desc="Création des clips...")

        resultats = []

        total = len(selection)

        for index, passage in enumerate(selection):

            debut = passage["debut"]
            fin = passage["fin"]

            nom = f"ClipIA_{index + 1}.mp4"

            sortie = os.path.join(
                dossier,
                nom
            )

            creer_clip(
                video,
                debut,
                fin,
                sortie,
                format_video
            )

            resultats.append(sortie)

            progression = 0.70 + (
                0.28 * (index + 1) / total
            )

            progress(
                progression,
                desc=f"Clip {index + 1}/{total} terminé..."
            )

        progress(1, desc="Terminé !")

        message = (
            f"✅ ClipIA a créé {len(resultats)} clip(s) !\n"
            f"Durée de la vidéo : {int(duree)} secondes.\n"
            f"Format : {format_video}."
        )

        return resultats, message

    except gr.Error:
        raise

    except subprocess.TimeoutExpired:
        raise gr.Error(
            "Le traitement a dépassé le temps autorisé. "
            "Essaie avec une vidéo plus courte."
        )

    except Exception as erreur:

        print("Erreur ClipIA :", repr(erreur))

        raise gr.Error(
            f"Une erreur est survenue : {str(erreur)}"
        )


# ============================================================
# INTERFACE GRADIO
# ============================================================

with gr.Blocks(
    title="ClipIA V3"
) as demo:

    gr.Markdown(
        """
        # 🎬 ClipIA V3

        ### Transforme automatiquement tes vidéos en clips !

        Ajoute une vidéo, choisis le nombre de clips et laisse
        ClipIA analyser les meilleurs passages.
        """
    )

    with gr.Row():

        video = gr.Video(
            label="Choisis ta vidéo",
            sources=["upload"],
            type="filepath"
        )

    with gr.Row():

        nombre = gr.Slider(
            minimum=1,
            maximum=5,
            value=3,
            step=1,
            label="Nombre de clips"
        )

        format_video = gr.Radio(
            choices=[
                "Original",
                "Vertical 9:16"
            ],
            value="Vertical 9:16",
            label="Format des clips"
        )

    bouton = gr.Button(
        "🚀 Générer mes clips",
        variant="primary"
    )

    resultat = gr.Gallery(
        label="Tes clips ClipIA",
        columns=2,
        height="auto",
        object_fit="contain"
    )

    message = gr.Textbox(
        label="État du traitement",
        interactive=False
    )

    bouton.click(
        fn=analyser_video,
        inputs=[
            video,
            nombre,
            format_video
        ],
        outputs=[
            resultat,
            message
        ]
    )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":

    demo.queue(
        max_size=5
    ).launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 7860)
        ),
        theme=gr.themes.Soft()
        )
