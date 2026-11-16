import os
import re
import cv2
import uuid
import tempfile
import subprocess
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
from faster_whisper import WhisperModel


# ============================================================
# CONFIGURATION CLIPIA
# ============================================================

APP_NAME = "ClipIA V8"
MAX_CLIPS = 5
CLIP_DURATION = 45

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")

OUTPUT_ROOT = Path(tempfile.gettempdir()) / "clipia_outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_MODEL = None


# ============================================================
# OUTILS
# ============================================================

def obtenir_modele():
    global _MODEL

    if _MODEL is None:
        _MODEL = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8"
        )

    return _MODEL


def lancer_commande(command):

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        message = (
            result.stderr[-3500:]
            if result.stderr
            else "Erreur FFmpeg inconnue."
        )

        raise RuntimeError(message)

    return result


def duree_video(path):

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError(
            "Impossible d'ouvrir cette vidéo. Essaie un fichier MP4."
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    capture.release()

    if not fps or fps <= 0:
        raise ValueError(
            "Impossible de déterminer la durée de la vidéo."
        )

    duration = frames / fps

    if duration <= 0:
        raise ValueError("La vidéo semble vide.")

    return duration


def extraire_audio(video_path, audio_path):

    lancer_commande([
        _FFMPEG,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path)
    ])


def nettoyer_texte(text):

    text = re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip()

    return text


# ============================================================
# MOTS-CLÉS BUSINESS
# ============================================================

MOTS_BUSINESS = {
    "business", "entreprise", "client", "clients",
    "vente", "vendre", "marketing", "revenu",
    "revenus", "chiffre", "argent", "investir",
    "investissement", "entrepreneur", "entrepreneuriat",
    "startup", "stratégie", "strategie", "équipe",
    "equipe", "manager", "management", "croissance",
    "profit", "bénéfice", "benefice", "marque",
    "offre", "produit", "marché", "marche",
    "objectif", "résultat", "resultat", "réussite",
    "reussite", "échec", "echec", "prix",
    "négocier", "negocier", "compétence",
    "competence", "travail", "carrière", "carriere"
}


MOTS_ACCROCHE = {
    "pourquoi", "comment", "attention", "jamais",
    "toujours", "erreur", "secret", "important",
    "problème", "probleme", "solution", "conseil",
    "apprendre", "comprendre", "découvrir",
    "decouvrir", "incroyable", "exemple",
    "meilleur", "meilleure", "pire", "vrai",
    "vérité", "verite"
}


def score_segment(text):

    lower = nettoyer_texte(text).lower()

    words = re.findall(
        r"[a-zà-ÿ']+",
        lower
    )

    if not words:
        return 0.0

    business = sum(
        1 for word in words
        if word in MOTS_BUSINESS
    )

    accroche = sum(
        1 for word in words
        if word in MOTS_ACCROCHE
    )

    question = 1.5 if "?" in text else 0.0

    longueur = min(
        len(words) / 18.0,
        2.0
    )

    return (
        business * 2.0
        + accroche * 1.2
        + question
        + longueur
    )


# ============================================================
# TRANSCRIPTION WHISPER
# ============================================================

def transcrire_audio(audio_path, progress):

    progress(
        0.25,
        desc="Chargement du modèle de transcription…"
    )

    model = obtenir_modele()

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=1,
        vad_filter=True,
        word_timestamps=False
    )

    resultat = []

    for segment in segments:

        text = nettoyer_texte(segment.text)

        if text:

            resultat.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "score": score_segment(text)
            })

    if not resultat:

        raise ValueError(
            "Aucune parole détectée. "
            "Vérifie que la vidéo contient une voix audible."
        )

    return resultat, getattr(
        info,
        "language",
        "inconnue"
    )


# ============================================================
# SÉLECTION AUTOMATIQUE DES PASSAGES
# ============================================================

def choisir_passages(segments, video_duration, nombre):

    clip_len = min(
        float(CLIP_DURATION),
        float(video_duration)
    )

    candidats = []

    for segment in segments:

        center = (
            segment["start"] + segment["end"]
        ) / 2

        start = max(
            0.0,
            center - clip_len / 2
        )

        end = min(
            video_duration,
            start + clip_len
        )

        start = max(
            0.0,
            end - clip_len
        )

        voisinage = [
            item
            for item in segments
            if (
                item["end"] >= start
                and item["start"] <= end
            )
        ]

        score = sum(
            item["score"]
            for item in voisinage
        )

        texte = " ".join(
            item["text"]
            for item in voisinage
        )

        if score > 0:

            candidats.append({
                "start": start,
                "end": end,
                "score": score,
                "text": texte
            })

    if not candidats:

        if video_duration <= clip_len:

            return [{
                "start": 0.0,
                "end": video_duration,
                "score": 1.0
            }]

        step = max(
            clip_len,
            video_duration / max(1, nombre)
        )

        return [
            {
                "start": min(
                    i * step,
                    max(0, video_duration - clip_len)
                ),
                "end": min(
                    i * step + clip_len,
                    video_duration
                ),
                "score": 1.0
            }
            for i in range(nombre)
        ]

    candidats.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selection = []

    for candidate in candidats:

        overlap = False

        for chosen in selection:

            intersection = max(
                0.0,
                min(
                    candidate["end"],
                    chosen["end"]
                )
                - max(
                    candidate["start"],
                    chosen["start"]
                )
            )

            if intersection > clip_len * 0.45:

                overlap = True
                break

        if not overlap:
            selection.append(candidate)

        if len(selection) >= nombre:
            break

    if len(selection) < nombre:

        for candidate in candidats:

            if candidate not in selection:
                selection.append(candidate)

            if len(selection) >= nombre:
                break

    selection.sort(
        key=lambda item: item["start"]
    )

    return selection[:nombre]


# ============================================================
# SOUS-TITRES
# ============================================================

def format_srt_time(seconds):

    milliseconds = int(
        max(0, seconds) * 1000
    )

    hours = milliseconds // 3600000
    milliseconds %= 3600000

    minutes = milliseconds // 60000
    milliseconds %= 60000

    secs = milliseconds // 1000
    millis = milliseconds % 1000

    return (
        f"{hours:02}:"
        f"{minutes:02}:"
        f"{secs:02},"
        f"{millis:03}"
    )


def emoji_business(text):

    lower = text.lower()

    if any(
        word in lower
        for word in (
            "argent",
            "revenu",
            "prix",
            "vente",
            "profit"
        )
    ):
        return "💰 "

    if any(
        word in lower
        for word in (
            "client",
            "marketing",
            "marque",
            "vente"
        )
    ):
        return "📈 "

    if any(
        word in lower
        for word in (
            "équipe",
            "equipe",
            "manager",
            "management"
        )
    ):
        return "🤝 "

    if any(
        word in lower
        for word in (
            "objectif",
            "stratégie",
            "strategie",
            "réussite",
            "reussite"
        )
    ):
        return "🎯 "

    return "💡 "


def creer_srt(
    segments,
    clip_start,
    clip_end,
    srt_path,
    emojis
):

    lignes = []
    index = 1

    for segment in segments:

        start = max(
            segment["start"],
            clip_start
        )

        end = min(
            segment["end"],
            clip_end
        )

        if end <= start:
            continue

        text = nettoyer_texte(
            segment["text"]
        )

        if emojis:
            text = emoji_business(text) + text

        local_start = start - clip_start
        local_end = end - clip_start

        lignes.append(str(index))

        lignes.append(
            f"{format_srt_time(local_start)} --> "
            f"{format_srt_time(local_end)}"
        )

        lignes.append(text)
        lignes.append("")

        index += 1

    srt_path.write_text(
        "\n".join(lignes),
        encoding="utf-8"
    )


def chemin_filtre_ffmpeg(path):

    value = str(
        path.resolve()
    ).replace("\\", "\\\\")

    value = value.replace(":", "\\:")
    value = value.replace("'", "\\'")
    value = value.replace(",", "\\,")
    value = value.replace("[", "\\[")
    value = value.replace("]", "\\]")

    return value


# ============================================================
# FORMATS VIDÉO
# ============================================================

def filtre_format(format_video):

    if format_video == "Vertical 9:16":

        return (
            "scale=1080:1920:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1920"
        )

    if format_video == "Carré 1:1":

        return (
            "scale=1080:1080:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1080"
        )

    if format_video == "Horizontal 16:9":

        return (
            "scale=1920:1080:"
            "force_original_aspect_ratio=increase,"
            "crop=1920:1080"
        )

    return None


# ============================================================
# FABRICATION DES CLIPS
# ============================================================

def fabriquer_clip(
    video_path,
    passage,
    index,
    dossier,
    segments,
    format_video,
    sous_titres,
    emojis
):

    start = max(
        0.0,
        passage["start"]
    )

    duration = max(
        0.5,
        passage["end"] - passage["start"]
    )

    output_path = (
        dossier / f"ClipIA_clip_{index:02d}.mp4"
    )

    filters = []

    video_filter = filtre_format(
        format_video
    )

    if video_filter:
        filters.append(video_filter)

    if sous_titres:

        srt_path = (
            dossier / f"clip_{index:02d}.srt"
        )

        creer_srt(
            segments,
            start,
            start + duration,
            srt_path,
            emojis
        )

        filters.append(
            "subtitles='"
            + chemin_filtre_ffmpeg(srt_path)
            + "':force_style='"
            "FontName=DejaVu Sans,"
            "FontSize=18,"
            "Outline=2,"
            "Shadow=0,"
            "Alignment=2,"
            "MarginV=45'"
        )

    command = [
        _FFMPEG,
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "0:a?"
    ]

    if filters:

        command += [
            "-vf",
            ",".join(filters)
        ]

    command += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ]

    lancer_commande(command)

    if (
        not output_path.exists()
        or output_path.stat().st_size < 10000
    ):

        raise RuntimeError(
            f"Le clip {index} n'a pas été créé correctement."
        )

    return output_path


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def generer_clips(
    video,
    nombre_clips,
    format_video,
    sous_titres_actifs,
    emojis_actifs,
    progress=gr.Progress()
):

    if not video:

        raise gr.Error(
            "Ajoute d'abord une vidéo de podcast."
        )

    video_path = Path(video)

    if not video_path.exists():

        raise gr.Error(
            "Le fichier vidéo est introuvable. Réimporte-le."
        )

    try:

        progress(
            0.03,
            desc="Vérification de la vidéo…"
        )

        video_duration = duree_video(
            video_path
        )

        job_id = uuid.uuid4().hex[:10]

        dossier = OUTPUT_ROOT / job_id
        dossier.mkdir(
            parents=True,
            exist_ok=True
        )

        audio_path = dossier / "audio.wav"

        progress(
            0.10,
            desc="Extraction de la piste audio…"
        )

        extraire_audio(
            video_path,
            audio_path
        )

        progress(
            0.20,
            desc="Transcription du podcast…"
        )

        segments, langue = transcrire_audio(
            audio_path,
            progress
        )

        nombre = max(
            1,
            min(MAX_CLIPS, int(nombre_clips))
        )

        progress(
            0.55,
            desc="Sélection des meilleurs passages…"
        )

        passages = choisir_passages(
            segments,
            video_duration,
            nombre
        )

        fichiers = []

        total = len(passages)

        for index, passage in enumerate(
            passages,
            start=1
        ):

            base = (
                0.60
                + (
                    0.38
                    * (index - 1)
                    / max(1, total)
                )
            )

            progress(
                base,
                desc=f"Création du clip {index}/{total}…"
            )

            output_path = fabriquer_clip(
                video_path,
                passage,
                index,
                dossier,
                segments,
                format_video,
                bool(sous_titres_actifs),
                bool(emojis_actifs)
            )

            fichiers.append(
                str(output_path)
            )

        try:

            audio_path.unlink(
                missing_ok=True
            )

        except Exception:
            pass

        progress(
            1.0,
            desc="Terminé !"
        )

        status = (
            f"✅ {len(fichiers)} clip(s) généré(s) !\n"
            f"🗣️ Langue détectée : {langue}\n"
            f"⏱️ Durée du podcast : "
            f"{int(video_duration // 60)} min "
            f"{int(video_duration % 60)} sec\n"
            f"🎬 Format : {format_video}\n"
            f"💬 Sous-titres : "
            f"{'activés' if sous_titres_actifs else 'désactivés'}\n"
            f"✨ Émojis business : "
            f"{'activés' if emojis_actifs else 'désactivés'}\n\n"
            "Télécharge tes clips dans la liste ci-dessous."
        )

        return fichiers, status

    except gr.Error:
        raise

    except Exception as error:

        raise gr.Error(
            "La génération a échoué.\n\n"
            f"Détail : {str(error)[-1800:]}\n\n"
            "Vérifie que la vidéo contient une piste audio "
            "et réessaie."
        )


# ============================================================
# INTERFACE GRADIO
# ============================================================

CSS = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#titre {
    text-align: center;
}

#description {
    text-align: center;
    opacity: 0.85;
}
"""


with gr.Blocks(
    title=APP_NAME
) as application:

    gr.Markdown(
        """
        # 🎬 ClipIA V8
        ### Transforme automatiquement tes podcasts en clips courts.
        """,
        elem_id="titre"
    )

    gr.Markdown(
        """
        Ajoute une vidéo, choisis tes options et laisse ClipIA
        sélectionner les passages intéressants.
        """,
        elem_id="description"
    )

    with gr.Row():

        with gr.Column():

            video = gr.Video(
                label="📁 Vidéo du podcast",
                sources=["upload"],
                type="filepath"
            )

            nombre_clips = gr.Slider(
                minimum=1,
                maximum=MAX_CLIPS,
                value=3,
                step=1,
                label="Nombre de clips à générer"
            )

            format_video = gr.Dropdown(
                choices=[
                    "Original",
                    "Vertical 9:16",
                    "Carré 1:1",
                    "Horizontal 16:9"
                ],
                value="Vertical 9:16",
                label="Format des clips"
            )

        with gr.Column():

            gr.Markdown(
                "### ⚙️ Personnalisation"
            )

            sous_titres_actifs = gr.Checkbox(
                value=True,
                label="💬 Activer les sous-titres"
            )

            emojis_actifs = gr.Checkbox(
                value=True,
                label="✨ Activer les émojis business"
            )

            gr.Markdown(
                """
                Les sous-titres et les émojis sont deux options séparées.
                Les émojis sont ajoutés aux sous-titres lorsqu'ils sont activés.
                """
            )

    bouton = gr.Button(
        "🚀 GÉNÉRER MES CLIPS",
        variant="primary",
        size="lg"
    )

    statut = gr.Textbox(
        label="📊 Progression et résultat",
        interactive=False,
        lines=8
    )

    fichiers = gr.File(
        label="🎬 Tes clips",
        file_count="multiple",
        interactive=False
    )

    bouton.click(
        fn=generer_clips,
        inputs=[
            video,
            nombre_clips,
            format_video,
            sous_titres_actifs,
            emojis_actifs
        ],
        outputs=[
            fichiers,
            statut
        ]
    )

    gr.Markdown(
        """
        ---
        **ClipIA V8** · Génération automatique de clips de podcast
        """
    )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":

    application.queue()

    application.launch(
        server_name="0.0.0.0",
        server_port=int(
   
