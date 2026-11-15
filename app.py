
import os
import re
import cv2
import time
import math
import shutil
import tempfile
import subprocess
import gradio as gr
import imageio_ffmpeg

from faster_whisper import WhisperModel

# ============================================================
# CLIPIA V6 — CONFIGURATION GÉNÉRALE
# ============================================================

APP_NAME = "ClipIA V6"

NOMBRE_MAX_CLIPS = 5
DUREE_MIN_CLIP = 10
DUREE_MAX_CLIP = 30

DUREE_TRANCHE_AUDIO = 600
DEBIT_AUDIO = "64k"

DOSSIER_SORTIE = os.path.join(
    tempfile.gettempdir(),
    "clipia_v6"
)

os.makedirs(DOSSIER_SORTIE, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

MODELE_WHISPER = None

FORMATS = [
    "Original",
    "Vertical 9:16",
    "Carré 1:1",
    "Horizontal 16:9"
]

# ============================================================
# EMOJIS BUSINESS
# ============================================================

MOTS_BUSINESS = {
    "business": "💼",
    "entreprise": "🏢",
    "argent": "💰",
    "million": "💰",
    "millions": "💰",
    "investissement": "📈",
    "investir": "📈",
    "investisseur": "📈",
    "marketing": "📢",
    "client": "🤝",
    "clients": "🤝",
    "vente": "🛒",
    "vendre": "🛒",
    "stratégie": "🎯",
    "strategie": "🎯",
    "objectif": "🎯",
    "objectifs": "🎯",
    "succès": "🏆",
    "succes": "🏆",
    "réussite": "🏆",
    "reussite": "🏆",
    "équipe": "👥",
    "equipe": "👥",
    "travail": "💻",
    "entrepreneur": "🚀",
    "entrepreneuriat": "🚀",
    "croissance": "📈",
    "chiffre": "📊",
    "résultat": "📊",
    "resultat": "📊",
    "temps": "⏱️",
    "productivité": "⚡",
    "productivite": "⚡",
    "idée": "💡",
    "idee": "💡",
    "projet": "🚀",
    "podcast": "🎙️",
    "communication": "🎤",
    "risque": "⚠️",
    "important": "⭐",
    "attention": "⚠️",
    "conseil": "💡",
    "conseils": "💡",
    "problème": "⚠️",
    "probleme": "⚠️",
    "solution": "✅",
    "formation": "🎓",
    "apprendre": "📚",
    "apprentissage": "📚",
    "innovation": "💡",
    "argent": "💰",
    "revenu": "💰",
    "revenus": "💰",
    "vente": "🛒",
    "vendre": "🛒"
}

# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def executer_commande(commande):
    resultat = subprocess.run(
        commande,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if resultat.returncode != 0:
        erreur = resultat.stderr or "Erreur FFmpeg inconnue."
        raise RuntimeError(erreur[-3000:])

    return resultat


def nettoyer_dossier():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    for nom in os.listdir(DOSSIER_SORTIE):
        chemin = os.path.join(DOSSIER_SORTIE, nom)

        try:
            if os.path.isfile(chemin):
                os.remove(chemin)
            elif os.path.isdir(chemin):
                shutil.rmtree(chemin)
        except Exception:
            pass


def verifier_video(video):
    if isinstance(video, dict):
        video = video.get("path")

    if not video:
        raise gr.Error("Ajoute une vidéo.")

    if not os.path.isfile(video):
        raise gr.Error("Le fichier vidéo est introuvable.")

    return video


# ============================================================
# CHARGEMENT WHISPER
# ============================================================

def charger_modele():
    global MODELE_WHISPER

    if MODELE_WHISPER is None:
        MODELE_WHISPER = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=2
        )

    return MODELE_WHISPER


# ============================================================
# INFORMATIONS VIDÉO
# ============================================================

def obtenir_infos_video(video):
    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise RuntimeError(
            "Impossible d'ouvrir la vidéo."
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    nombre_frames = capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    largeur = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    hauteur = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    capture.release()

    if fps <= 0 or nombre_frames <= 0:
        raise RuntimeError(
            "Les informations de la vidéo sont invalides."
        )

    duree = nombre_frames / fps

    return {
        "fps": fps,
        "frames": nombre_frames,
        "largeur": largeur,
        "hauteur": hauteur,
        "duree": duree
    }


# ============================================================
# EXTRACTION AUDIO COMPRESSÉE
# ============================================================

def extraire_audio(video, destination):
    executer_commande([
        FFMPEG,
        "-y",
        "-i", video,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "libmp3lame",
        "-b:a", DEBIT_AUDIO,
        destination
    ])


# ============================================================
# TRANSCRIPTION PAR TRANCHES DE 10 MINUTES
# ============================================================

def transcrire_audio(
    audio,
    duree_video,
    progress=None
):
    modele = charger_modele()

    tous_les_segments = []

    nombre_tranches = max(
        1,
        math.ceil(
            duree_video / DUREE_TRANCHE_AUDIO
        )
    )

    for numero in range(nombre_tranches):

        debut_tranche = (
            numero * DUREE_TRANCHE_AUDIO
        )

        duree_restante = (
            duree_video - debut_tranche
        )

        duree_actuelle = min(
            DUREE_TRANCHE_AUDIO,
            duree_restante
        )

        if duree_actuelle <= 0:
            break

        fichier_tranche = os.path.join(
            DOSSIER_SORTIE,
            f"tranche_{numero + 1}.mp3"
        )

        if progress:
            avancement = (
                0.22
                + 0.32 * (
                    numero / nombre_tranches
                )
            )

            progress(
                avancement,
                desc=(
                    f"Transcription "
                    f"{numero + 1}/{nombre_tranches} "
                    f"— {int(debut_tranche / 60)} min"
                )
            )

        executer_commande([
            FFMPEG,
            "-y",
            "-ss", str(debut_tranche),
            "-i", audio,
            "-t", str(duree_actuelle),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "libmp3lame",
            "-b:a", DEBIT_AUDIO,
            fichier_tranche
        ])

        segments, informations = modele.transcribe(
            fichier_tranche,
            beam_size=1,
            vad_filter=True,
            language="fr"
        )

        for segment in segments:
            texte = segment.text.strip()

            if not texte:
                continue

            tous_les_segments.append({
                "debut": (
                    float(segment.start)
                    + debut_tranche
                ),
                "fin": (
                    float(segment.end)
                    + debut_tranche
                ),
                "texte": texte
            })

        try:
            os.remove(fichier_tranche)
        except Exception:
            pass

    return tous_les_segments


# ============================================================
# ANALYSE DES TEXTES
# ============================================================

def calculer_score(texte):
    texte_min = texte.lower()

    mots_importants = [
        "business",
        "argent",
        "million",
        "investissement",
        "entreprise",
        "succès",
        "réussite",
        "stratégie",
        "marketing",
        "client",
        "vente",
        "entrepreneur",
        "objectif",
        "résultat",
        "secret",
        "conseil",
        "erreur",
        "problème",
        "solution",
        "important",
        "incroyable",
        "jamais",
        "toujours",
        "apprendre",
        "travail",
        "croissance",
        "productivité"
    ]

    score = 0

    for mot in mots_importants:
        if mot in texte_min:
            score += 2

    if "?" in texte:
        score += 3

    if "!" in texte:
        score += 2

    if len(texte) > 100:
        score += 2

    if len(texte) > 200:
        score += 2

    return score


# ============================================================
# CONSTRUCTION DES PASSAGES CANDIDATS
# ============================================================

def construire_candidats(
    segments,
    duree_video
):
    candidats = []

    if not segments:
        return candidats

    for index, segment_depart in enumerate(segments):

        debut = segment_depart["debut"]

        textes = []
        score = 0
        fin = debut

        for segment in segments[index:]:

            if segment["debut"] - debut > DUREE_MAX_CLIP:
                break

            if (
                segment["debut"] - fin > 5
                and textes
            ):
                break

            textes.append(segment["texte"])

            score += calculer_score(
                segment["texte"]
            )

            fin = segment["fin"]

            duree = fin - debut

            if duree >= DUREE_MIN_CLIP:

                if duree > DUREE_MAX_CLIP:
                    break

                candidats.append({
                    "debut": max(0, debut),
                    "fin": min(duree_video, fin),
                    "score": score,
                    "texte": " ".join(textes)
                })

                break

    return candidats


# ============================================================
# SÉLECTION DES MEILLEURS CLIPS
# ============================================================

def selectionner_clips(
    candidats,
    duree_video,
    nombre
):
    selection = []

    candidats = sorted(
        candidats,
        key=lambda x: x["score"],
        reverse=True
    )

    for candidat in candidats:

        debut = candidat["debut"]
        fin = candidat["fin"]

        chevauchement = False

        for choisi in selection:

            if not (
                fin <= choisi["debut"]
                or debut >= choisi["fin"]
            ):
                chevauchement = True
                break

        if not chevauchement:
            selection.append(candidat)

        if len(selection) >= nombre:
            break

    # Compléter si Whisper n'a pas trouvé assez
    # de passages suffisamment longs.

    if len(selection) < nombre:

        duree_clip = min(
            DUREE_MAX_CLIP,
            duree_video
        )

        if duree_video > duree_clip:
            espace = (
                duree_video - duree_clip
            ) / max(nombre, 1)
        else:
            espace = 0

        for index in range(nombre):

            debut = index * espace

            fin = min(
                duree_video,
                debut + duree_clip
            )

            if fin - debut < DUREE_MIN_CLIP:
                continue

            nouveau = {
                "debut": debut,
                "fin": fin,
                "score": 0,
                "texte": ""
            }

            chevauchement = any(
                not (
                    fin <= choisi["debut"]
                    or debut >= choisi["fin"]
                )
                for choisi in selection
            )

            if not chevauchement:
                selection.append(nouveau)

            if len(selection) >= nombre:
                break

    return sorted(
        selection,
        key=lambda x: x["debut"]
    )[:nombre]


# ============================================================
# SOUS-TITRES ASS
# ============================================================

def temps_ass(secondes):
    secondes = max(0, secondes)

    heures = int(secondes // 3600)

    minutes = int(
        (secondes % 3600) // 60
    )

    secondes_entieres = int(
        secondes % 60
    )

    centiemes = int(
        (secondes - int(secondes)) * 100
    )

    return (
        f"{heures}:"
        f"{minutes:02d}:"
        f"{secondes_entieres:02d}."
        f"{centiemes:02d}"
    )


def echapper_ass(texte):
    texte = texte.replace("\\", "\\\\")
    texte = texte.replace("{", "\\{")
    texte = texte.replace("}", "\\}")
    texte = texte.replace("\n", " ")

    return texte


def trouver_emoji(texte):
    texte_min = texte.lower()

    for mot, emoji in MOTS_BUSINESS.items():

        expression = (
            r"\b"
            + re.escape(mot)
            + r"\b"
        )

        if re.search(expression, texte_min):
            return emoji

    return None


def creer_fichier_ass(
    segments,
    clip,
    destination,
    largeur,
    hauteur,
    sous_titres_actifs,
    emojis_actifs
):
    taille = max(
        24,
        int(hauteur * 0.027)
    )

    marge = max(
        35,
        int(hauteur * 0.055)
    )

    contenu = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {largeur}
PlayResY: {hauteur}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: SousTitre,DejaVu Sans,{taille},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,1,3,1,2,45,45,{marge},1
Style: Emoji,DejaVu Sans,{int(taille * 1.8)},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,1,2,0,8,30,30,{marge},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    debut_clip = clip["debut"]
    fin_clip = clip["fin"]

    for segment in segments:

        if (
            segment["fin"] <= debut_clip
            or segment["debut"] >= fin_clip
        ):
            continue

        debut = max(
            0,
            segment["debut"] - debut_clip
        )

        fin = min(
            fin_clip - debut_clip,
            segment["fin"] - debut_clip
        )

        if fin <= debut:
            continue

        texte = echapper_ass(
            segment["texte"]
        )

        if sous_titres_actifs:

            contenu += (
                f"Dialogue: 0,"
                f"{temps_ass(debut)},"
                f"{temps_ass(fin)},"
                f"SousTitre,,0,0,0,,"
                f"{texte}\n"
            )

        if emojis_actifs:

            emoji = trouver_emoji(
                segment["texte"]
            )

            if emoji:

                contenu += (
                    f"Dialogue: 1,"
                    f"{temps_ass(debut)},"
                    f"{temps_ass(fin)},"
                    f"Emoji,,0,0,0,,"
                    f"{emoji}\n"
                )

    with open(
        destination,
        "w",
        encoding="utf-8"
    ) as fichier:

        fichier.write(contenu)


# ============================================================
# FORMATS VIDÉO
# ============================================================

def obtenir_filtre_format(
    format_video,
    largeur_originale,
    hauteur_originale
):
    if format_video == "Vertical 9:16":

        largeur = 1080
        hauteur = 1920

        filtre = (
            "scale=1080:1920:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1920"
        )

    elif format_video == "Carré 1:1":

        largeur = 1080
        hauteur = 1080

        filtre = (
            "scale=1080:1080:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1080"
        )

    elif format_video == "Horizontal 16:9":

        largeur = 1920
        hauteur = 1080

        filtre = (
            "scale=1920:1080:"
            "force_original_aspect_ratio=increase,"
            "crop=1920:1080"
        )

    else:

        largeur = largeur_originale
        hauteur = hauteur_originale

        if largeur % 2:
            largeur -= 1

        if hauteur % 2:
            hauteur -= 1

        filtre = (
            f"scale={largeur}:{hauteur}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={largeur}:{hauteur}:"
            "(ow-iw)/2:(oh-ih)/2"
        )

    return filtre, largeur, hauteur


# ============================================================
# FILTRE SOUS-TITRES
# ============================================================

def preparer_chemin_ass(chemin):
    chemin = os.path.abspath(chemin)

    chemin = chemin.replace("\\", "/")
    chemin = chemin.replace(":", "\\:")
    chemin = chemin.replace("'", "\\'")

    return chemin


# ============================================================
# CRÉATION DES CLIPS
# ============================================================

def creer_clip(
    video,
    infos,
    clip,
    numero,
    format_video,
    sous_titres_actifs,
    emojis_actifs,
    segments
):
    debut = clip["debut"]

    duree = (
        clip["fin"] - clip["debut"]
    )

    destination = os.path.join(
        DOSSIER_SORTIE,
        f"ClipIA_V6_{numero}.mp4"
    )

    filtre, largeur, hauteur = (
        obtenir_filtre_format(
            format_video,
            infos["largeur"],
            infos["hauteur"]
        )
    )

    filtres = [filtre]

    if (
        sous_titres_actifs
        or emojis_actifs
    ):

        fichier_ass = os.path.join(
            DOSSIER_SORTIE,
            f"sous_titres_{numero}.ass"
        )

        creer_fichier_ass(
            segments,
            clip,
            fichier_ass,
            largeur,
            hauteur,
            sous_titres_actifs,
            emojis_actifs
        )

        chemin_ass = preparer_chemin_ass(
            fichier_ass
        )

        filtres.append(
            f"ass='{chemin_ass}'"
        )

    filtre_final = ",".join(filtres)

    commande = [
        FFMPEG,
        "-y",
        "-ss", str(debut),
        "-i", video,
        "-t", str(duree),
        "-vf", filtre_final,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        destination
    ]

    executer_commande(commande)

    if not os.path.exists(destination):
        raise RuntimeError(
            "Le clip n'a pas été créé."
        )

    if os.path.getsize(destination) == 0:
        raise RuntimeError(
            "Le fichier vidéo créé est vide."
        )

    return destination


# ============================================================
# GÉNÉRATION PRINCIPALE
# ============================================================

def generer_clips(
    video,
    nombre_clips,
    format_video,
    sous_titres_actifs,
    emojis_actifs,
    progress=gr.Progress()
):
    debut_traitement = time.time()

    video = verifier_video(video)

    nombre_clips = int(nombre_clips)

    nombre_clips = max(
        1,
        min(
            NOMBRE_MAX_CLIPS,
            nombre_clips
        )
    )

    nettoyer_dossier()

    try:

        # --------------------------------------------
        # ÉTAPE 1 : INFORMATIONS VIDÉO
        # --------------------------------------------

        progress(
            0.03,
            desc="Analyse de la vidéo..."
        )

        infos = obtenir_infos_video(video)

        if infos["duree"] < DUREE_MIN_CLIP:

            raise gr.Error(
                "La vidéo est trop courte."
            )

        # --------------------------------------------
        # ÉTAPE 2 : EXTRACTION AUDIO
        # --------------------------------------------

        progress(
            0.10,
            desc="Extraction audio compressée..."
        )

  
    
        
    
        
    

        
            
