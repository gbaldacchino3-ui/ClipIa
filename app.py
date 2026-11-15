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
# CLIPIA V5 — CONFIGURATION
# ============================================================

NOMBRE_MAX_CLIPS = 5
DUREE_MIN_CLIP = 10
DUREE_MAX_CLIP = 30

DOSSIER_SORTIE = os.path.join(tempfile.gettempdir(), "clipia_v5")
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

MODELE_WHISPER = None

FORMATS = {
    "Original": "original",
    "Vertical 9:16": "vertical",
    "Carré 1:1": "carre",
    "Horizontal 16:9": "horizontal"
}

MOTS_BUSINESS = {
    "business": "💼",
    "entreprise": "🏢",
    "argent": "💰",
    "million": "💰",
    "investissement": "📈",
    "investir": "📈",
    "marketing": "📢",
    "client": "🤝",
    "vente": "🛒",
    "vendre": "🛒",
    "stratégie": "🎯",
    "strategie": "🎯",
    "objectif": "🎯",
    "succès": "🏆",
    "succes": "🏆",
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
    "attention": "⚠️"
}

# ============================================================
# WHISPER
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
# OUTILS
# ============================================================

def executer_commande(command):
    resultat = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if resultat.returncode != 0:
        raise RuntimeError(resultat.stderr[-2500:])

    return resultat


def obtenir_infos_video(fichier):
    capture = cv2.VideoCapture(fichier)

    if not capture.isOpened():
        raise ValueError("Impossible de lire cette vidéo.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    largeur = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.release()

    if fps <= 0:
        raise ValueError("FPS vidéo invalide.")

    duree = frames / fps

    return {
        "duree": duree,
        "fps": fps,
        "largeur": largeur,
        "hauteur": hauteur
    }


def extraire_audio(video, destination):
    executer_commande([
        FFMPEG,
        "-y",
        "-i", video,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        destination
    ])


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcrire_audio(audio):
    modele = charger_modele()

    segments, informations = modele.transcribe(
        audio,
        beam_size=1,
        vad_filter=True
    )

    resultat = []

    for segment in segments:
        texte = segment.text.strip()

        if texte:
            resultat.append({
                "debut": float(segment.start),
                "fin": float(segment.end),
                "texte": texte
            })

    return resultat


# ============================================================
# SÉLECTION DES PASSAGES
# ============================================================

def calculer_score(texte):
    texte_min = texte.lower()
    score = 0

    mots_importants = [
        "mais", "pourquoi", "comment", "résultat",
        "argent", "business", "million", "erreur",
        "réussite", "succès", "stratégie", "important",
        "incroyable", "jamais", "toujours", "secret",
        "conseil", "apprendre", "entreprise",
        "investissement", "client", "problème"
    ]

    for mot in mots_importants:
        if mot in texte_min:
            score += 2

    if "?" in texte:
        score += 3

    if "!" in texte:
        score += 2

    if len(texte) > 100:
        score += 2

    return score


def construire_candidats(segments, duree_video):
    candidats = []

    if not segments:
        return candidats

    index = 0

    while index < len(segments):
        debut = segments[index]["debut"]
        fin = debut
        textes = []
        score = 0

        j = index

        while j < len(segments):
            segment = segments[j]

            nouvelle_fin = segment["fin"]

            if nouvelle_fin - debut > DUREE_MAX_CLIP:
                break

            if nouvelle_fin - debut >= DUREE_MIN_CLIP:
                if j > index and (
                    segment["debut"] - fin > 3
                ):
                    break

            textes.append(segment["texte"])
            score += calculer_score(segment["texte"])
            fin = nouvelle_fin
            j += 1

            if fin - debut >= DUREE_MIN_CLIP:
                texte_complet = " ".join(textes)

                candidats.append({
                    "debut": max(0, debut),
                    "fin": min(duree_video, fin),
                    "score": score,
                    "texte": texte_complet
                })

                break

        index = max(index + 1, j)

    return candidats


def selectionner_clips(candidats, duree_video, nombre):
    candidats = sorted(
        candidats,
        key=lambda x: x["score"],
        reverse=True
    )

    selection = []

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

    # Si Whisper n'a pas trouvé assez de passages,
    # on complète avec des segments répartis dans la vidéo.

    if len(selection) < nombre:
        duree_clip = min(
            DUREE_MAX_CLIP,
            max(DUREE_MIN_CLIP, duree_video / max(nombre, 1))
        )

        positions = []

        if duree_video > duree_clip:
            espace = (duree_video - duree_clip) / max(nombre, 1)

            for i in range(nombre):
                positions.append(i * espace)
        else:
            positions.append(0)

        for position in positions:
            debut = max(0, position)
            fin = min(duree_video, debut + duree_clip)

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
                    fin <= x["debut"]
                    or debut >= x["fin"]
                )
                for x in selection
            )

            if not chevauchement:
                selection.append(nouveau)

            if len(selection) >= nombre:
                break

    return sorted(selection, key=lambda x: x["debut"])[:nombre]


# ============================================================
# SOUS-TITRES ET EMOJIS
# ============================================================

def temps_ass(secondes):
    heures = int(secondes // 3600)
    minutes = int((secondes % 3600) // 60)
    secondes_entieres = int(secondes % 60)
    centiemes = int((secondes - int(secondes)) * 100)

    return (
        f"{heures}:{minutes:02d}:"
        f"{secondes_entieres:02d}.{centiemes:02d}"
    )


def echapper_ass(texte):
    texte = texte.replace("\\", "\\\\")
    texte = texte.replace("{", "\\{")
    texte = texte.replace("}", "\\}")
    texte = texte.replace("\n", " ")
    return texte


def trouver_emoji(texte):
    texte = texte.lower()

    for mot, emoji in MOTS_BUSINESS.items():
        if re.search(r"\b" + re.escape(mot) + r"\b", texte):
            return emoji

    return None


def creer_fichier_ass(
    segments,
    debut_clip,
    fin_clip,
    destination,
    largeur,
    hauteur,
    sous_titres_actifs,
    emojis_actifs
):
    taille = max(24, int(hauteur * 0.027))
    marge = max(35, int(hauteur * 0.055))

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

    for segment in segments:
        debut = max(0, segment["debut"] - debut_clip)
        fin = min(
            fin_clip - debut_clip,
            segment["fin"] - debut_clip
        )

        if fin <= 0 or debut >= fin_clip - debut_clip:
            continue

        debut = max(0, debut)
        fin = max(debut + 0.2, fin)

        texte = echapper_ass(segment["texte"])

        if sous_titres_actifs:
            contenu += (
                f"Dialogue: 0,{temps_ass(debut)},"
                f"{temps_ass(fin)},SousTitre,,0,0,0,,"
                f"{texte}\n"
            )

        if emojis_actifs:
            emoji = trouver_emoji(segment["texte"])

            if emoji:
                contenu += (
                    f"Dialogue: 1,{temps_ass(debut)},"
                    f"{temps_ass(fin)},Emoji,,0,0,0,,"
                    f"{emoji}\n"
                )

    with open(destination, "w", encoding="utf-8") as fichier:
        fichier.write(contenu)


# ============================================================
# FORMAT VIDÉO
# ============================================================

def filtre_format(format_video, largeur_originale, hauteur_originale):
    if format_video == "Vertical 9:16":
        largeur, hauteur = 1080, 1920
        filtre = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920"
        )

    elif format_video == "Carré 1:1":
        largeur, hauteur = 1080, 1080
        filtre = (
            "scale=1080:1080:force_original_aspect_ratio=increase,"
            "crop=1080:1080"
        )

    elif format_video == "Horizontal 16:9":
        largeur, hauteur = 1920, 1080
        filtre = (
            "scale=1920:1080:force_original_aspect_ratio=increase,"
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
            f"pad={largeur}:{hauteur}:(ow-iw)/2:(oh-ih)/2"
        )

    return filtre, largeur, hauteur


# ============================================================
# CRÉATION D'UN CLIP
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
    duree = clip["fin"] - clip["debut"]

    nom = f"ClipIA_V5_{numero}.mp4"
    destination = os.path.join(DOSSIER_SORTIE, nom)

    filtre, largeur, hauteur = filtre_format(
        format_video,
        infos["largeur"],
        infos["hauteur"]
    )

    filtres = [filtre]

    if sous_titres_actifs or emojis_actifs:
        fichier_ass = os.path.join(
            DOSSIER_SORTIE,
            f"sous_titres_{numero}.ass"
        )

        creer_fichier_ass(
            segments,
            debut,
            clip["fin"],
            fichier_ass,
            largeur,
            hauteur,
            sous_titres_actifs,
            emojis_actifs
        )

        chemin_ass = fichier_ass.replace("\\", "/")
        chemin_ass = chemin_ass.replace(":", "\\:")

        filtres.append(f"ass='{chemin_ass}'")

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

    return destination


# ============================================================
# TRAITEMENT PRINCIPAL
# ============================================================

def generer_clips(
    video,
    nombre_clips,
    format_video,
    sous_titres_actifs,
    emojis_actifs,
    progress=gr.Progress()
):
    if video is None:
        raise gr.Error("Ajoute une vidéo avant de commencer.")

    if isinstance(video, dict):
        video = video.get("path")

    if not video or not os.path.exists(video):
        raise gr.Error("Fichier vidéo introuvable.")

    nombre_clips = int(nombre_clips)

    nombre_clips = max(
        1,
        min(NOMBRE_MAX_CLIPS, nombre_clips)
    )

    for fichier in os.listdir(DOSSIER_SORTIE):
        chemin = os.path.join(DOSSIER_SORTIE, fichier)

        try:
            if os.path.isfile(chemin):
                os.remove(chemin)
            elif os.path.isdir(chemin):
                shutil.rmtree(chemin)
        except Exception:
            pass

    try:
        progress(0.05, desc="Analyse de la vidéo...")

        infos = obtenir_infos_video(video)

        if infos["duree"] < DUREE_MIN_CLIP:
            raise gr.Error(
                "La vidéo est trop courte pour créer un clip."
            )

        progress(0.12, desc="Extraction de l'audio...")

        audio = os.path.join(
            DOSSIER_SORTIE,
            "audio_temp.wav"
        )

        extraire_audio(video, audio)

        progress(0.25, desc="Transcription avec Whisper...")

        segments = transcrire_audio(audio)

        progress(0.55, desc="Recherche des meilleurs passages...")

        candidats = construire_candidats(
            segments,
            infos["duree"]
        )

        clips = selectionner_clips(
            candidats,
            infos["duree"],
            nombre_clips
        )

        if not clips:
            raise gr.Error(
                "Aucun passage exploitable n'a été trouvé."
            )

        resultats = []

        for index, clip in enumerate(clips):
            progression = 0.60 + (
                0.38 * (index / max(len(clips), 1))
            )

            progress(
                progression,
                desc=f"Création du clip {index + 1}/{len(clips)}..."
            )

            resultat = creer_clip(
                video,
                infos,
                clip,
                index + 1,
                format_video,
                sous_titres_actifs,
                emojis_actifs,
                segments
            )

            resultats.append(resultat)

        progress(1, desc="Terminé !")

        try:
            os.remove(audio)
        except Exception:
            pass

        return resultats, (
            f"✅ {len(resultats)} clip(s) créé(s) avec succès !"
        )

    except gr.Error:
        raise

    except Exception as erreur:
        raise gr.Error(
            "Une erreur est survenue :\n"
            + str(erreur)[-1800:]
        )


# ============================================================
# INTERFACE CLIPIA V5
# ============================================================

CSS = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}

#titre {
    text-align: center;
    margin-bottom: 5px;
}

#description {
    text-align: center;
    opacity: 0.8;
}

#generer {
    background: #16a34a !important;
    color: white !important;
    font-weight: bold !important;
    min-height: 55px;
    font-size: 18px !important;
}

#statut {
    text-align: center;
    font-weight: bold;
}
"""

with gr.Blocks() as application:

    gr.Markdown(
        "# 🎬 ClipIA V5",
        elem_id="titre"
    )

    gr.Markdown(
        "Transforme automatiquement tes podcasts en clips courts.",
        elem_id="description"
    )

    with gr.Row():
        video = gr.Video(
            label="📁 Importer une vidéo",
            sources=["upload"]
        )

    with gr.Row():
        nombre_clips = gr.Slider(
            minimum=1,
            maximum=NOMBRE_MAX_CLIPS,
            value=3,
            step=1,
            label="Nombre de clips à créer"
        )

        format_video = gr.Dropdown(
            choices=list(FORMATS.keys()),
            value="Vertical 9:16",
            label="Format vidéo"
        )

    gr.Markdown("### Personnalisation")

    with gr.Row():
        sous_titres_actifs = gr.Checkbox(
            value=True,
            label="📝 Activer les sous-titres"
        )

        emojis_actifs = gr.Checkbox(
            value=True,
            label="💼 Activer les emojis business"
        )

    generer = gr.Button(
        "🚀 GÉNÉRER MES CLIPS",
        elem_id="generer"
    )

    statut = gr.Markdown(
        "En attente d'une vidéo.",
        elem_id="statut"
    )

    resultats = gr.Files(
        label="🎥 Tes clips terminés"
    )

    generer.click(
        fn=generer_clips,
        inputs=[
            video,
            nombre_clips,
            format_video,
            sous_titres_actifs,
            emojis_actifs
        ],
        outputs=[
            resultats,
            statut
        ]
    )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    application.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
        css=CSS
    )
    
