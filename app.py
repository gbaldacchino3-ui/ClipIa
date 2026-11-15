import os
import re
import cv2
import time
import math
import tempfile
import subprocess
import gradio as gr
import imageio_ffmpeg

from faster_whisper import WhisperModel

# ============================================================
# CLIPIA V5 — CONFIGURATION
# ============================================================

NOMBRE_MAX_CLIPS = 5
DUREE_MAX_CLIP = 30
DUREE_MIN_CLIP = 10
INTERVALLE_ANALYSE = 4

DOSSIER_SORTIE = os.path.join(
    tempfile.gettempdir(), "clipia_v5"
)
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

FORMATS = {
    "Original": None,
    "Vertical 9:16": (720, 1280),
    "Carré 1:1": (1080, 1080),
    "Horizontal 16:9": (1280, 720)
}

# ============================================================
# WHISPER
# ============================================================

print("Chargement de Whisper...")

modele_whisper = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
    cpu_threads=2
)

print("Whisper V5 prêt !")


# ============================================================
# OUTILS
# ============================================================

def nettoyer_dossier():
    for fichier in os.listdir(DOSSIER_SORTIE):
        chemin = os.path.join(DOSSIER_SORTIE, fichier)
        try:
            if os.path.isfile(chemin):
                os.remove(chemin)
        except Exception:
            pass


def chemin_video(video):
    if isinstance(video, dict):
        video = video.get("path") or video.get("name")

    if not video or not os.path.isfile(video):
        raise gr.Error("Fichier vidéo introuvable.")

    return video


def obtenir_infos_video(video):

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise gr.Error("Impossible d'ouvrir la vidéo.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    images = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    largeur = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    hauteur = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    capture.release()

    if fps <= 0:
        raise gr.Error("FPS vidéo invalides.")

    duree = images / fps

    if duree <= 0:
        raise gr.Error("Durée vidéo invalide.")

    return {
        "fps": fps,
        "duree": duree,
        "largeur": int(largeur),
        "hauteur": int(hauteur)
    }


# ============================================================
# SOUS-TITRES ET EMOJIS BUSINESS
# ============================================================

MOTS_EMOJIS = {
    "argent": "💰",
    "revenu": "💰",
    "revenus": "💰",
    "million": "💰",
    "millions": "💰",
    "investissement": "📈",
    "investir": "📈",
    "croissance": "📈",
    "business": "💼",
    "entreprise": "💼",
    "entrepreneur": "🚀",
    "entrepreneuriat": "🚀",
    "objectif": "🎯",
    "stratégie": "🧠",
    "intelligence": "🧠",
    "apprendre": "💡",
    "idée": "💡",
    "attention": "⚠️",
    "erreur": "⚠️",
    "important": "💡",
    "succès": "🏆",
    "réussite": "🏆",
    "partenaire": "🤝",
    "partenariat": "🤝",
    "travail": "💼",
    "équipe": "🤝",
    "croître": "📈",
    "marketing": "📣",
    "vente": "💰",
    "vendre": "💰",
    "client": "🤝"
}


def trouver_emoji(texte):

    texte = texte.lower()

    for mot, emoji in MOTS_EMOJIS.items():
        if re.search(
            r"\b" + re.escape(mot) + r"\b",
            texte
        ):
            return emoji

    return None


def nettoyer_texte_ass(texte):
    """Échappe les caractères spéciaux ASS."""

    texte = texte.replace("\\", "\\\\")
    texte = texte.replace("{", "(")
    texte = texte.replace("}", ")")
    texte = texte.replace("\n", " ")

    return texte


def temps_ass(secondes):

    secondes = max(0, secondes)

    heures = int(secondes // 3600)
    minutes = int((secondes % 3600) // 60)
    secondes_entieres = int(secondes % 60)
    centiemes = int((secondes % 1) * 100)

    return (
        f"{heures}:{minutes:02d}:"
        f"{secondes_entieres:02d}."
        f"{centiemes:02d}"
    )


def creer_sous_titres(audio, debut, fin, numero, activer_emoji):

    fichier_ass = os.path.join(
        DOSSIER_SORTIE,
        f"sous_titres_{numero}.ass"
    )

    dialogues = []

    for segment in audio:

        if segment["fin"] <= debut or segment["debut"] >= fin:
            continue

        debut_local = max(0, segment["debut"] - debut)
        fin_local = min(fin - debut, segment["fin"] - debut)

        if fin_local <= debut_local:
            continue

        texte = nettoyer_texte_ass(segment["texte"])

        if not texte:
            continue

        # Sous-titres professionnels :
        # blanc avec contour noir.
        dialogues.append(
            "Dialogue: 0,"
            f"{temps_ass(debut_local)},"
            f"{temps_ass(fin_local)},"
            "SousTitres,,0,0,0,,"
            f"{texte}"
        )

        # Emoji séparé du texte.
        if activer_emoji:

            emoji = trouver_emoji(segment["texte"])

            if emoji:
                dialogues.append(
                    "Dialogue: 1,"
                    f"{temps_ass(debut_local)},"
                    f"{temps_ass(fin_local)},"
                    "Emojis,,0,0,0,,"
                    f"{emoji}"
                )

    contenu = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: SousTitres,Arial,65,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,70,70,180,1
Style: Emojis,Arial,90,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,8,70,70,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    contenu += "\n".join(dialogues)

    with open(fichier_ass, "w", encoding="utf-8") as fichier:
        fichier.write(contenu)

    return fichier_ass


# ============================================================
# ANALYSE AUDIO
# ============================================================

def analyser_audio(video, progress):

    progress(0.05, desc="Analyse de la voix...")

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

        return resultats

    except Exception as erreur:

        print("Erreur Whisper :", erreur)

        return []


# ============================================================
# ANALYSE VISUELLE
# ============================================================

def analyser_visuellement(video, informations, progress):

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise gr.Error("Impossible d'analyser les images.")

    duree = informations["duree"]

    nombre_points = max(
        1,
        int(duree / INTERVALLE_ANALYSE)
    )

    resultats = []
    image_precedente = None

    for index in range(nombre_points + 1):

        temps = min(
            index * INTERVALLE_ANALYSE,
            duree
        )

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            temps * 1000
        )

        succes, image = capture.read()

        if not succes:
            continue

        image = cv2.resize(
            image,
            (320, 180)
        )

        gris = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        nettete = cv2.Laplacian(
            gris,
            cv2.CV_64F
        ).var()

        contraste = gris.std()
        luminosite = gris.mean()

        score = min(nettete / 10, 30)
        score += min(contraste / 3, 25)

        if 50 < luminosite < 210:
            score += 20

        if image_precedente is not None:

            difference = cv2.absdiff(
                gris,
                image_precedente
            )

            mouvement = difference.mean()

            score += min(mouvement * 1.5, 25)

        resultats.append({
            "temps": temps,
            "score": score
        })

        image_precedente = gris

        progress(
            0.25 + 0.25 * (index + 1) / (nombre_points + 1),
            desc=f"Analyse visuelle {index + 1}/{nombre_points + 1}"
        )

    capture.release()

    return resultats


# ============================================================
# SCORE AUDIO
# ============================================================

def calculer_score_audio(texte):

    texte = texte.lower()

    mots = [
        "incroyable", "impossible", "pourquoi",
        "comment", "attention", "important",
        "argent", "million", "réussite",
        "business", "stratégie", "jamais",
        "énorme", "surprise", "problème",
        "solution", "objectif", "résultat",
        "investissement", "croissance"
    ]

    score = 0

    for mot in mots:
        if re.search(r"\b" + re.escape(mot) + r"\b", texte):
            score += 8

    if "!" in texte:
        score += 5

    if "?" in texte:
        score += 3

    if len(texte.split()) >= 8:
        score += 5

    return score


def trouver_score_visuel(temps, analyses):

    if not analyses:
        return 0

    resultat = min(
        analyses,
        key=lambda element: abs(element["temps"] - temps)
    )

    return resultat["score"]


def construire_moments(audio, visuel):

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

        moments.append({
            "debut": segment["debut"],
            "fin": segment["fin"],
            "milieu": milieu,
            "score": score_audio + score_visuel,
            "texte": segment["texte"]
        })

    if not moments:

        for element in visuel:

            moments.append({
                "debut": element["temps"],
                "fin": element["temps"],
                "milieu": element["temps"],
                "score": element["score"],
                "texte": ""
            })

    return sorted(
        moments,
        key=lambda element: element["score"],
        reverse=True
    )


# ============================================================
# SÉLECTION DES CLIPS
# ============================================================

def selectionner_clips(moments, duree_video, nombre_clips):

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

        trop_proche = any(
            abs(centre - selection["milieu"])
            < DUREE_MAX_CLIP * 0.75
            for selection in selections
        )

        if trop_proche:
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

    # Complément si nécessaire

    if len(selections) < nombre_clips:

        nombre = max(1, nombre_clips)

        for index in range(nombre):

            if len(selections) >= nombre_clips:
                break

            centre = (
                duree_video * (index + 0.5) / nombre
            )

            if any(
                abs(centre - clip["milieu"])
                < DUREE_MAX_CLIP * 0.75
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
# CRÉATION VIDÉO
# ============================================================

def creer_clip(
    video,
    debut,
    fin,
    numero,
    format_video,
    audio,
    activer_sous_titres,
    activer_emoji
):

    sortie = os.path.join(
        DOSSIER_SORTIE,
        f"ClipIA_V5_{numero}.mp4"
    )

    duree = fin - debut

    commande = [
        FFMPEG,
        "-y",
        "-ss", str(debut),
        "-i", video,
        "-t", str(duree)
    ]

    filtres = []

    dimensions = FORMATS[format_video]

    if dimensions:

        largeur, hauteur = dimensions

        filtres.append(
            f"scale={largeur}:{hauteur}:"
            "force_original_aspect_ratio=decrease"
        )

        filtres.append(
            f"pad={largeur}:{hauteur}:(ow-iw)/2:(oh-ih)/2"
        )

    if activer_sous_titres:

        fichier_ass = creer_sous_titres(
            audio,
            debut,
            fin,
            numero,
            activer_emoji
        )

        # Échappement du chemin pour le filtre FFmpeg
        chemin_ass = os.path.abspath(
            fichier_ass
        ).replace("\\", "/")

        chemin_ass = chemin_ass.replace(":", "\\:")
        chemin_ass = chemin_ass.replace("'", "\\'")

        filtres.append(
            f"ass='{chemin_ass}'"
        )

    if filtres:

        commande += [
            "-vf", ",".join(filtres)
        ]

    commande += [
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "26",
        "-threads", "2",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-max_muxing_queue_size", "1024",
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
            resultat.stderr[-2500:]
        )

    if (
        not os.path.exists(sortie)
        or os.path.getsize(sortie) == 0
    ):
        raise RuntimeError(
            "Le fichier vidéo est vide."
        )

    return sortie


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def generer_clips(
    video,
    nombre_clips,
    format_video,
    activer_sous_titres,
    activer_emoji,
    progress=gr.Progress()
):

    if video is None:
        raise gr.Error("Ajoute une vidéo.")

    debut_total = time.time()

    try:

        video = chemin_video(video)

        nombre_clips = int(nombre_clips)

        if not 1 <= nombre_clips <= NOMBRE_MAX_CLIPS:
            raise gr.Error("Choisis entre 1 et 5 clips.")

        if format_video not in FORMATS:
            raise gr.Error("Format vidéo invalide.")

        nettoyer_dossier()

        progress(
            0.02,
            desc="Lecture de la vidéo..."
        )

        informations = obtenir_infos_video(video)

        duree = informations["duree"]

        if duree < DUREE_MIN_CLIP:
            raise gr.Error("La vidéo est trop courte.")

        # 1. AUDIO

        audio = analyser_audio(
            video,
            progress
        )

        # 2. VISUEL

        visuel = analyser_visuellement(
            video,
            informations,
            progress
        )

        # 3. SÉLECTION

        progress(
            0.55,
            desc="Recherche des meilleurs passages..."
        )

        moments = construire_moments(
            audio,
            visuel
        )

        selections = selectionner_clips(
            moments,
            duree,
            nombre_clips
        )

        if not selections:
            raise gr.Error("Aucun clip exploitable.")

        # 4. CRÉATION

        fichiers = []
        descriptions = []

        for index, selection in enumerate(selections):

            progress(
                0.60 + 0.38 * index / len(selections),
                desc=f"Création du clip {index + 1}/{len(selections)}..."
            )

            fichier = creer_clip(
                video,
                selection["debut"],
                selection["fin"],
                index + 1,
                format_video,
                audio,
                activer_sous_titres,
                activer_emoji
            )

            fichiers.append(fichier)

            description = (
                f"Clip {index + 1} : "
                f"{selection['debut']:.1f}s - "
                f"{selection['fin']:.1f}s"
            )

            if selection["texte"]:
                description += (
                    "\nPassage : " + selection["texte"]
                )

            descriptions.append(description)

        temps_total = time.time() - debut_total

        progress(
            1,
            desc="ClipIA V5 terminé !"
        )

        message = (
            f"🎉 {len(fichiers)} clip(s) créé(s) !\n"
            f"Temps : {temps_total:.1f} secondes\n"
            f"Format : {format_video}\n"
            f"Sous-titres : {'Oui' if activer_sous_titres else 'Non'}\n"
            f"Emojis : {'Oui' if activer_emoji else 'Non'}"
        )

        return (
            fichiers,
            message,
            "\n\n".join(descriptions)
        )

    except gr.Error:
        raise

    except subprocess.TimeoutExpired:
        raise gr.Error(
            "Le traitement a dépassé le délai autorisé."
        )

    except Exception as erreur:

        print(
            "ERREUR CLIPIA V5 :",
            repr(erreur)
        )

        raise gr.Error(
            f"Erreur : {str(erreur)[:1500]}"
        )


# ============================================================
# INTERFACE
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
    min-height: 58px;
    font-size: 19px;
    font-weight: bold;
}

#resultat {
    border-radius: 15px;
}
"""


with gr.Blocks(
    css=CSS,
    theme=gr.themes.Soft()
) as application:

    gr.Markdown(
        """
        # 🎬 CLIPIA V5

        ### Ton studio de création automatique de clips business.

        Transforme tes podcasts en clips professionnels.
        """,
        elem_id="titre"
    )

    with gr.Row():

        with gr.Column():

            video = gr.Video(
                label="📹 Importer ton podcast",
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
                choices=list(FORMAT
