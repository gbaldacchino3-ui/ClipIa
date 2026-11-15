
import os
import re
import cv2
import time
import shutil
import tempfile
import subprocess
import gradio as gr
import imageio_ffmpeg

from faster_whisper import WhisperModel

# ============================================================
# CLIPIA — CONFIGURATION
# ============================================================

NOMBRE_MAX_CLIPS = 5
DUREE_MAX_CLIP = 30
DUREE_MIN_CLIP = 10
INTERVALLE_ANALYSE = 4

DOSSIER_SORTIE = os.path.join(tempfile.gettempdir(), "clipia_output")
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Chargement du modèle une seule fois
print("Chargement de Whisper...")
modele_whisper = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
    cpu_threads=2
)
print("Whisper est prêt !")


# ============================================================
# OUTILS
# ============================================================

def nettoyer_dossier():
    """Nettoie les anciens clips générés."""
    try:
        for fichier in os.listdir(DOSSIER_SORTIE):
            chemin = os.path.join(DOSSIER_SORTIE, fichier)
            if os.path.isfile(chemin):
                os.remove(chemin)
    except Exception:
        pass


def obtenir_infos_video(video):
    """Récupère les informations principales de la vidéo."""

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise gr.Error("Impossible d'ouvrir cette vidéo.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    nombre_images = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    largeur = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    hauteur = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    capture.release()

    if not fps or fps <= 0:
        raise gr.Error("Impossible de déterminer les FPS de la vidéo.")

    duree = nombre_images / fps if nombre_images else 0

    if duree <= 0:
        raise gr.Error("La durée de la vidéo est invalide.")

    return {
        "fps": fps,
        "duree": duree,
        "largeur": int(largeur),
        "hauteur": int(hauteur),
        "nombre_images": int(nombre_images)
    }


# ============================================================
# ANALYSE AUDIO AVEC WHISPER
# ============================================================

def analyser_audio(video, progress=gr.Progress()):
    """Transcrit l'audio et récupère les passages intéressants."""

    progress(0.05, desc="Analyse de l'audio...")

    try:
        segments, informations = modele_whisper.transcribe(
            video,
            language="fr",
            beam_size=1,
            best_of=1,
            vad_filter=True,
            condition_on_previous_text=False
        )

        resultats = []

        for segment in segments:
            texte = segment.text.strip()

            if not texte:
                continue

            resultats.append({
                "debut": float(segment.start),
                "fin": float(segment.end),
                "texte": texte
            })

        progress(0.25, desc="Analyse audio terminée")

        return resultats

    except Exception as erreur:
        print("Erreur Whisper :", erreur)
        return []


# ============================================================
# ANALYSE VISUELLE
# ============================================================

def analyser_visuellement(video, informations, progress=gr.Progress()):
    """
    Analyse des images à intervalles réguliers.
    Utilise une recherche directe par timestamp pour éviter
    de lire inutilement toutes les images.
    """

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise gr.Error("Impossible d'analyser les images.")

    duree = informations["duree"]
    resultats = []

    nombre_points = max(1, int(duree / INTERVALLE_ANALYSE))

    for index in range(nombre_points + 1):

        temps = min(index * INTERVALLE_ANALYSE, duree)

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            temps * 1000
        )

        succes, image = capture.read()

        if not succes or image is None:
            continue

        image_reduite = cv2.resize(
            image,
            (320, 180)
        )

        gris = cv2.cvtColor(
            image_reduite,
            cv2.COLOR_BGR2GRAY
        )

        nettete = cv2.Laplacian(
            gris,
            cv2.CV_64F
        ).var()

        contraste = gris.std()

        luminosite = gris.mean()

        score = 0

        # Netteté
        score += min(nettete / 10, 30)

        # Contraste
        score += min(contraste / 3, 25)

        # Luminosité
        if 50 < luminosite < 210:
            score += 20

        # Mouvement approximatif entre images
        if resultats:
            difference = cv2.absdiff(
                gris,
                resultats[-1]["image"]
            )

            mouvement = difference.mean()
            score += min(mouvement * 1.5, 25)

        resultats.append({
            "temps": temps,
            "score": score,
            "image": gris
        })

        progress(
            0.25 + 0.25 * (index + 1) / nombre_points,
            desc=f"Analyse visuelle : {index + 1}/{nombre_points}"
        )

    capture.release()

    # On ne conserve pas les images en mémoire
    for resultat in resultats:
        resultat.pop("image", None)

    return resultats


# ============================================================
# ANALYSE DES MOMENTS INTÉRESSANTS
# ============================================================

def calculer_score_audio(texte):
    """Attribue un score selon les mots présents dans la transcription."""

    texte = texte.lower()

    mots_interessants = [
        "incroyable",
        "impossible",
        "attends",
        "quoi",
        "pourquoi",
        "comment",
        "haha",
        "mdr",
        "wow",
        "oh",
        "non",
        "oui",
        "sérieux",
        "jamais",
        "énorme",
        "attention",
        "regarde",
        "surprise",
        "problème",
        "réussi",
        "raté",
        "drôle",
        "mort de rire"
    ]

    score = 0

    for mot in mots_interessants:
        if mot in texte:
            score += 8

    if "!" in texte:
        score += 5

    if "?" in texte:
        score += 3

    return score


def trouver_score_visuel(temps, analyses):
    """Trouve le score visuel le plus proche d'un instant."""

    if not analyses:
        return 0

    resultat = min(
        analyses,
        key=lambda element: abs(element["temps"] - temps)
    )

    return resultat["score"]


def construire_moments(audio, visuel, duree):
    """Combine les analyses audio et visuelle."""

    moments = []

    for segment in audio:

        milieu = (
            segment["debut"] + segment["fin"]
        ) / 2

        score_audio = calculer_score_audio(
            segment["texte"]
        )

        score_visuel = trouver_score_visuel(
            milieu,
            visuel
        )

        score_total = score_audio + score_visuel

        moments.append({
            "debut": segment["debut"],
            "fin": segment["fin"],
            "milieu": milieu,
            "score": score_total,
            "texte": segment["texte"]
        })

    # Si Whisper n'a trouvé aucun texte, on utilise l'analyse visuelle
    if not moments:
        for element in visuel:
            moments.append({
                "debut": element["temps"],
                "fin": element["temps"],
                "milieu": element["temps"],
                "score": element["score"],
                "texte": ""
            })

    moments.sort(
        key=lambda element: element["score"],
        reverse=True
    )

    return moments


# ============================================================
# SÉLECTION DES CLIPS
# ============================================================

def selectionner_clips(moments, duree_video, nombre_clips):
    """Sélectionne les meilleurs moments sans trop de chevauchement."""

    selections = []

    for moment in moments:

        centre = moment["milieu"]

        debut = max(
            0,
            centre - DUREE_MAX_CLIP / 2
        )

        fin = min(
            duree_video,
            debut + DUREE_MAX_CLIP
        )

        debut = max(
            0,
            fin - DUREE_MAX_CLIP
        )

        if fin - debut < DUREE_MIN_CLIP:
            continue

        # Évite les clips trop proches les uns des autres
        chevauchement = False

        for selection in selections:

            distance = abs(
                centre - selection["milieu"]
            )

            if distance < DUREE_MAX_CLIP * 0.75:
                chevauchement = True
                break

        if chevauchement:
            continue

        selections.append({
            "debut": debut,
            "fin": fin,
            "milieu": centre,
            "score": moment["score"],
            "texte": moment["texte"]
        })

        if len(selections) >= nombre_clips:
            break

    # Complète si nécessaire avec des moments espacés
    if len(selections) < nombre_clips:

        pas = max(
            DUREE_MAX_CLIP,
            duree_video / max(nombre_clips, 1)
        )

        for index in range(nombre_clips):

            if len(selections) >= nombre_clips:
                break

            centre = min(
                duree_video - DUREE_MIN_CLIP / 2,
                pas * index + pas / 2
            )

            if centre < 0:
                continue

            if any(
                abs(centre - clip["milieu"]) < DUREE_MAX_CLIP * 0.75
                for clip in selections
            ):
                continue

            debut = max(
                0,
                centre - DUREE_MAX_CLIP / 2
            )

            fin = min(
                duree_video,
                debut + DUREE_MAX_CLIP
            )

            debut = max(
                0,
                fin - DUREE_MAX_CLIP
            )

            if fin - debut >= DUREE_MIN_CLIP:
                selections.append({
                    "debut": debut,
                    "fin": fin,
                    "milieu": centre,
                    "score": 0,
                    "texte": ""
                })

    selections.sort(
        key=lambda element: element["debut"]
    )

    return selections[:nombre_clips]


# ============================================================
# CRÉATION DES CLIPS AVEC FFMPEG
# ============================================================

def creer_clip(video, debut, fin, numero, format_video):
    """Crée un clip avec FFmpeg."""

    sortie = os.path.join(
        DOSSIER_SORTIE,
        f"ClipIA_{numero}.mp4"
    )

    duree = max(1, fin - debut)

    commande = [
        FFMPEG,
        "-y",
        "-ss", str(debut),
        "-i", video,
        "-t", str(duree),
        "-analyzeduration", "10000000",
        "-probesize", "10000000"
    ]

    if format_video == "Original":

        commande += [
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            sortie
        ]

    else:

        filtre = (
            "scale=720:1280:"
            "force_original_aspect_ratio=decrease,"
            "pad=720:1280:(ow-iw)/2:(oh-ih)/2"
        )

        commande += [
            "-vf", filtre,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-threads", "2",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            sortie
        ]

    resultat = subprocess.run(
        commande,
        capture_output=True,
        text=True,
        timeout=300
    )

    if resultat.returncode != 0:
        raise RuntimeError(
            resultat.stderr[-2000:]
        )

    if not os.path.exists(sortie):
        raise RuntimeError("Le fichier vidéo n'a pas été créé.")

    if os.path.getsize(sortie) == 0:
        raise RuntimeError("Le fichier vidéo est vide.")

    return sortie


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def generer_clips(
    video,
    nombre_clips,
    format_video,
    progress=gr.Progress()
):
    """Lance toutes les étapes de création."""

    if video is None:
        raise gr.Error("Ajoute d'abord une vidéo.")

    try:
        nombre_clips = int(nombre_clips)

        if nombre_clips < 1 or nombre_clips > NOMBRE_MAX_CLIPS:
            raise gr.Error("Choisis entre 1 et 5 clips.")

        nettoyer_dossier()

        debut_total = time.time()

        progress(0.02, desc="Lecture de la vidéo...")

        informations = obtenir_infos_video(video)

        if informations["duree"] < DUREE_MIN_CLIP:
            raise gr.Error("La vidéo est trop courte.")

        # ----------------------------------------------------
        # ÉTAPE 1 : ANALYSE AUDIO
        # ----------------------------------------------------

        audio = analyser_audio(
            video,
            progress
        )

        # ----------------------------------------------------
        # ÉTAPE 2 : ANALYSE VISUELLE
        # ----------------------------------------------------

        visuel = analyser_visuellement(
            video,
            informations,
            progress
        )

        # ----------------------------------------------------
        # ÉTAPE 3 : ANALYSE DES MOMENTS
        # ----------------------------------------------------

        progress(
            0.55,
            desc="Recherche des meilleurs moments..."
        )

        moments = construire_moments(
            audio,
            visuel,
            informations["duree"]
        )

        selections = selectionner_clips(
            moments,
            informations["duree"],
            nombre_clips
        )

        if not selections:
            raise gr.Error(
                "Aucun moment exploitable n'a été trouvé."
            )

        # ----------------------------------------------------
        # ÉTAPE 4 : CRÉATION DES CLIPS
        # ----------------------------------------------------

        fichiers = []
        descriptions = []

        total = len(selections)

        for index, selection in enumerate(selections):

            progression = (
                0.60 + 0.38 * index / total
            )

            progress(
                progression,
                desc=f"Création du clip {index + 1}/{total}..."
            )

            try:
                fichier = creer_clip(
                    video,
                    selection["debut"],
                    selection["fin"],
                    index + 1,
                    format_video
                )

                fichiers.append(fichier)

                description = (
                    f"Clip {index + 1} — "
                    f"{selection['debut']:.1f}s à "
                    f"{selection['fin']:.1f}s"
                )

                if selection["texte"]:
                    description += (
                        f"\nTranscription : {selection['texte']}"
                    )

                descriptions.append(description)

            except Exception as erreur:
                print(
                    f"Erreur création clip {index + 1} :",
                    erreur
                )

        if not fichiers:
            raise gr.Error(
                "Aucun clip n'a pu être créé. Vérifie le format de la vidéo."
            )

        temps_total = time.time() - debut_total

        progress(
            1,
            desc="Création terminée !"
        )

        message = (
            f"Terminé ! {len(fichiers)} clip(s) créé(s).\n"
            f"Temps total : {temps_total:.1f} secondes.\n"
            f"Format : {format_video}"
        )

        return fichiers, message, "\n\n".join(descriptions)

    except gr.Error:
        raise

    except subprocess.TimeoutExpired:
        raise gr.Error(
            "Le traitement a dépassé le temps autorisé."
        )

    except Exception as erreur:
        print("ERREUR CLIPIA :", repr(erreur))

        raise gr.Error(
            f"Une erreur est survenue : {str(erreur)[:1000]}"
        )


# ============================================================
# INTERFACE GRADIO
# ============================================================

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#titre {
    text-align: center;
}

#bouton {
    min-height: 55px;
    font-size: 18px;
    font-weight: bold;
}

#resultat {
    border-radius: 15px;
}
"""


with gr.Blocks(css=CSS) as application:

    gr.Markdown(
        """
        # 🎬 ClipIA V4

        ### Transforme tes vidéos en clips automatiquement.

        Ajoute une vidéo, choisis le nombre de clips et laisse ClipIA
        rechercher les meilleurs moments.
        """,
        elem_id="titre"
    )

    with gr.Row():

        with gr.Column():

            video = gr.Video(
                label="Ta vidéo",
                sources=["upload"],
                format="mp4"
            )

            nombre_clips = gr.Slider(
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
                value="Original",
                label="Format des clips"
            )

            bouton = gr.Button(
                "🚀 GÉNÉRER MES CLIPS",
                variant="primary",
                elem_id="bouton"
            )

    gr.Markdown("## 📁 Résultats")

    message = gr.Textbox(
        label="État du traitement",
        interactive=False,
        lines=3
    )

    galerie = gr.File(
        label="Télécharger les clips",
        file_count="multiple",
        interactive=False
    )

    descriptions = gr.Textbox(
        label="Informations sur les clips",
        interactive=False,
        lines=8
    )

    bouton.click(
        fn=generer_clips,
        inputs=[
            video,
            nombre_clips,
            format_video
        ],
        outputs=[
            galerie,
            message,
            descriptions
        ]
    )

    gr.Markdown(
        """
        ---
        **ClipIA V4** — Génération automatique de clips.
        """
    )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":

    application.queue(
        default_concurrency_limit=1
    )

    application.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 7860)
        )
        )
        
