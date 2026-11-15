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
# CLIPIA V7 — CONFIGURATION
# ============================================================

NOMBRE_MAX_CLIPS = 5
DUREE_CLIP = 30
DUREE_MIN_CLIP = 10
DUREE_TRANCHE_AUDIO = 600

DOSSIER_TRAVAIL = "/tmp/clipia_v7"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

MODELE_WHISPER = None

os.makedirs(DOSSIER_TRAVAIL, exist_ok=True)

# ============================================================
# MOTS BUSINESS ET EMOJIS
# ============================================================

MOTS_BUSINESS = {
    "argent": "💰",
    "business": "💼",
    "entreprise": "🏢",
    "entrepreneur": "🚀",
    "entrepreneuriat": "🚀",
    "investissement": "📈",
    "investir": "📈",
    "million": "💎",
    "millionnaire": "💎",
    "marketing": "📢",
    "vente": "🤝",
    "vendre": "🤝",
    "client": "👥",
    "clients": "👥",
    "réussite": "🏆",
    "succès": "🏆",
    "objectif": "🎯",
    "travail": "💻",
    "équipe": "🤝",
    "croissance": "📈",
    "stratégie": "🧠",
    "idée": "💡",
    "innovation": "✨",
    "startup": "🚀",
    "risque": "⚡",
    "profit": "💰",
    "bénéfice": "💰",
    "chiffre": "📊",
    "chiffres": "📊",
    "podcast": "🎙️",
    "temps": "⏳",
    "formation": "🎓",
    "apprendre": "📚",
    "motivation": "🔥",
    "discipline": "💪",
    "liberté": "🦅",
    "indépendance": "🗝️",
}

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
        raise RuntimeError(
            resultat.stderr[-2500:]
        )

    return resultat


def nettoyer_nom(nom):
    nom = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        str(nom)
    )

    return nom[:60]


def nettoyer_dossier():
    try:
        if os.path.exists(DOSSIER_TRAVAIL):
            shutil.rmtree(DOSSIER_TRAVAIL)

        os.makedirs(
            DOSSIER_TRAVAIL,
            exist_ok=True
        )

    except Exception:
        pass


def verifier_video(video):
    if not video:
        raise gr.Error(
            "Ajoute une vidéo avant de lancer ClipIA."
        )

    if not os.path.exists(video):
        raise gr.Error(
            "Le fichier vidéo est introuvable."
        )

    return video


def obtenir_infos_video(video):
    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise gr.Error(
            "Impossible de lire cette vidéo."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    frames = capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    largeur = capture.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )

    hauteur = capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )

    capture.release()

    if not fps or fps <= 0:
        raise gr.Error(
            "Impossible de déterminer les FPS."
        )

    duree = frames / fps

    if duree < DUREE_MIN_CLIP:
        raise gr.Error(
            "La vidéo est trop courte."
        )

    return {
        "duree": duree,
        "fps": fps,
        "largeur": int(largeur),
        "hauteur": int(hauteur)
    }


# ============================================================
# CHARGEMENT WHISPER
# ============================================================

def charger_modele():
    global MODELE_WHISPER

    if MODELE_WHISPER is None:
        MODELE_WHISPER = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

    return MODELE_WHISPER


# ============================================================
# EXTRACTION AUDIO
# ============================================================

def extraire_audio(video, dossier):
    audio = os.path.join(
        dossier,
        "audio.mp3"
    )

    command = [
        FFMPEG,
        "-y",
        "-i", video,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "48k",
        audio
    ]

    executer_commande(command)

    if not os.path.exists(audio):
        raise RuntimeError(
            "L'extraction audio a échoué."
        )

    if os.path.getsize(audio) == 0:
        raise RuntimeError(
            "Le fichier audio est vide."
        )

    return audio


# ============================================================
# TRANSCRIPTION PAR TRANCHES
# ============================================================

def transcrire_audio(audio, duree, progress):
    modele = charger_modele()

    transcription = []

    nombre_tranches = max(
        1,
        int((duree + DUREE_TRANCHE_AUDIO - 1)
            // DUREE_TRANCHE_AUDIO)
    )

    for index in range(nombre_tranches):

        debut = index * DUREE_TRANCHE_AUDIO

        duree_tranche = min(
            DUREE_TRANCHE_AUDIO,
            duree - debut
        )

        if duree_tranche <= 0:
            break

        chemin_tranche = os.path.join(
            DOSSIER_TRAVAIL,
            f"tranche_{index}.mp3"
        )

        command = [
            FFMPEG,
            "-y",
            "-ss", str(debut),
            "-i", audio,
            "-t", str(duree_tranche),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "48k",
            chemin_tranche
        ]

        executer_commande(command)

        segments, informations = modele.transcribe(
            chemin_tranche,
            language="fr",
            beam_size=1,
            vad_filter=True,
            word_timestamps=False
        )

        for segment in segments:
            texte = segment.text.strip()

            if not texte:
                continue

            transcription.append({
                "start": float(segment.start) + debut,
                "end": float(segment.end) + debut,
                "text": texte
            })

        try:
            os.remove(chemin_tranche)
        except Exception:
            pass

        progression = (
            (index + 1) / nombre_tranches
        )

        progress(
            0.25 + progression * 0.35,
            desc=(
                f"Transcription : "
                f"{index + 1}/{nombre_tranches}"
            )
        )

    if not transcription:
        raise gr.Error(
            "Aucune parole détectée dans la vidéo."
        )

    return transcription


# ============================================================
# ANALYSE DES TRANSCRIPTIONS
# ============================================================

MOTS_IMPORTANTS = [
    "argent",
    "business",
    "entreprise",
    "million",
    "investissement",
    "investir",
    "réussite",
    "succès",
    "stratégie",
    "marketing",
    "client",
    "clients",
    "profit",
    "bénéfice",
    "objectif",
    "important",
    "incroyable",
    "problème",
    "solution",
    "erreur",
    "conseil",
    "apprendre",
    "résultat",
    "croissance",
    "liberté",
    "travail",
    "entrepreneur",
    "motivation",
    "discipline",
    "risque",
    "opportunité",
    "décision"
]


def calculer_score(texte):
    texte = texte.lower()

    score = 0

    for mot in MOTS_IMPORTANTS:
        if mot in texte:
            score += 3

    if "?" in texte:
        score += 2

    if "!" in texte:
        score += 2

    if any(
        mot in texte
        for mot in [
            "mais",
            "parce que",
            "pourquoi",
            "comment",
            "attention",
            "jamais",
            "toujours",
            "voici"
        ]
    ):
        score += 2

    nombre_mots = len(
        texte.split()
    )

    if nombre_mots >= 12:
        score += 2

    if nombre_mots >= 25:
        score += 2

    return score


# ============================================================
# CONSTRUCTION DES CANDIDATS
# ============================================================

def construire_candidats(transcription, duree):
    candidats = []

    for segment in transcription:

        milieu = (
            segment["start"] +
            segment["end"]
        ) / 2

        debut = max(
            0,
            milieu - DUREE_CLIP / 2
        )

        fin = min(
            duree,
            debut + DUREE_CLIP
        )

        if fin - debut < DUREE_MIN_CLIP:
            continue

        texte = segment["text"]

        score = calculer_score(texte)

        candidats.append({
            "start": debut,
            "end": fin,
            "score": score,
            "text": texte
        })

    candidats.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidats


# ============================================================
# SÉLECTION DES CLIPS
# ============================================================

def selectionner_clips(
    candidats,
    nombre,
    duree
):
    selection = []

    for candidat in candidats:

        if len(selection) >= nombre:
            break

        debut = candidat["start"]
        fin = candidat["end"]

        chevauchement = False

        for choisi in selection:

            if (
                debut < choisi["end"] + 8
                and fin > choisi["start"] - 8
            ):
                chevauchement = True
                break

        if not chevauchement:
            selection.append(candidat)

    # Compléter si nécessaire avec des positions espacées.
    if len(selection) < nombre:

        pas = max(
            DUREE_MIN_CLIP,
            duree / nombre
        )

        for index in range(nombre):

            if len(selection) >= nombre:
                break

            debut = max(
                0,
                min(
                    duree - DUREE_CLIP,
                    index * pas
                )
            )

            fin = min(
                duree,
                debut + DUREE_CLIP
            )

            chevauchement = any(
                debut < clip["end"] + 8
                and fin > clip["start"] - 8
                for clip in selection
            )

            if not chevauchement:
                selection.append({
                    "start": debut,
                    "end": fin,
                    "score": 0,
                    "text": ""
                })

    selection.sort(
        key=lambda x: x["start"]
    )

    return selection[:nombre]


# ============================================================
# GÉNÉRATION DES SOUS-TITRES
# ============================================================

def convertir_temps_srt(secondes):
    secondes = max(
        0,
        int(secondes * 1000)
    )

    heures = secondes // 3600000

    minutes = (
        secondes % 3600000
    ) // 60000

    secondes_reste = (
        secondes % 60000
    ) // 1000

    millisecondes = secondes % 1000

    return (
        f"{heures:02}:"
        f"{minutes:02}:"
        f"{secondes_reste:02},"
        f"{millisecondes:03}"
    )


def trouver_emoji(texte):
    texte = texte.lower()

    for mot, emoji in MOTS_BUSINESS.items():
        if mot in texte:
            return emoji

    return "🔥"


def creer_sous_titres(
    transcription,
    clip,
    chemin,
    sous_titres_actifs,
    emojis_actifs
):
    lignes = []

    numero = 1

    for segment in transcription:

        debut = max(
            segment["start"],
            clip["start"]
        )

        fin = min(
            segment["end"],
            clip["end"]
        )

        if fin <= debut:
            continue

        texte = segment["text"].strip()

        if not texte:
            continue

        contenu = ""

        if sous_titres_actifs:
            contenu = texte

        if emojis_actifs:
            emoji = trouver_emoji(texte)

            if contenu:
                contenu += " " + emoji
            else:
                contenu = emoji

        if not contenu:
            continue

        debut_local = debut - clip["start"]
        fin_local = fin - clip["start"]

        lignes.append(
            str(numero)
        )

        lignes.append(
            f"{convertir_temps_srt(debut_local)} --> "
            f"{convertir_temps_srt(fin_local)}"
        )

        lignes.append(contenu)
        lignes.append("")

        numero += 1

    with open(
        chemin,
        "w",
        encoding="utf-8"
    ) as fichier:
        fichier.write(
            "\n".join(lignes)
        )


# ============================================================
# FORMATS VIDÉO
# ============================================================

def obtenir_filtre_format(format_video):

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
# RENDU D'UN CLIP
# ============================================================

def generer_un_clip(
    video,
    clip,
    index,
    dossier,
    transcription,
    format_video,
    sous_titres_actifs,
    emojis_actifs
):
    sortie = os.path.join(
        dossier,
        f"ClipIA_{index + 1}.mp4"
    )

    filtres = []

    filtre_format = obtenir_filtre_format(
        format_video
    )

    if filtre_format:
        filtres.append(
            filtre_format
        )

    if (
        sous_titres_actifs
        or emojis_actifs
    ):
        chemin_srt = os.path.join(
            dossier,
            f"subtitles_{index}.srt"
        )

        creer_sous_titres(
            transcription,
            clip,
            chemin_srt,
            sous_titres_actifs,
            emojis_actifs
        )

        if os.path.getsize(chemin_srt) > 0:

            chemin_srt_filtre = (
                chemin_srt
                .replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("'", "\\'")
            )

            filtres.append(
                f"subtitles='{chemin_srt_filtre}'"
            )

    command = [
        FFMPEG,
        "-y",
        "-ss", str(clip["start"]),
        "-i", video,
        "-t", str(
            clip["end"] - clip["start"]
        ),
    ]

    if filtres:
        command.extend([
            "-vf",
            ",".join(filtres)
        ])

    command.extend([
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-shortest",
        sortie
    ])

    executer_commande(command)

    if (
        not os.path.exists(sortie)
        or os.path.getsize(sortie) == 0
    ):
        raise RuntimeError(
            f"Le clip {index + 1} n'a pas été créé."
        )

    return sortie


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
    dossier = None

    try:

        # ÉTAPE 1 : VÉRIFICATION

        video = verifier_video(video)

        dossier = tempfile.mkdtemp(
            prefix="clipia_",
            dir=DOSSIER_TRAVAIL
        )

        progress(
            0.03,
            desc="Analyse de la vidéo..."
        )

        infos = obtenir_infos_video(
            video
        )

        duree = infos["duree"]

        if duree < DUREE_MIN_CLIP:
            raise gr.Error(
                "La vidéo est trop courte."
            )

        # ÉTAPE 2 : EXTRACTION AUDIO

        progress(
            0.10,
            desc="Extraction audio compressée..."
        )

        audio = extraire_audio(
            video,
            dossier
        )

        # ÉTAPE 3 : TRANSCRIPTION

        progress(
            0.20,
            desc="Préparation de la transcription..."
        )

        transcription = transcrire_audio(
            audio,
            duree,
            progress
        )

        # ÉTAPE 4 : ANALYSE

        progress(
            0.62,
            desc="Analyse des meilleurs passages..."
        )

        candidats = construire_candidats(
            transcription,
            duree
        )

        if not candidats:
            raise gr.Error(
                "Impossible de trouver des passages exploitables."
            )

        nombre_clips = int(
            nombre_clips
        )

        nombre_clips = max(
            1,
            min(
                NOMBRE_MAX_CLIPS,
                nombre_clips
            )
        )

        clips = selectionner_clips(
            candidats,
            nombre_clips,
            duree
        )

        if not clips:
            raise gr.Error(
                "Aucun clip n'a pu être sélectionné."
            )

        # ÉTAPE 5 : RENDU

        resultats = []

        total = len(clips)

        for index, clip in enumerate(clips):

            progression = (
                index / total
            )

            progress(
                0.65 + progression * 0.32,
                desc=(
                    f"Création du clip "
                    f"{index + 1}/{total}..."
                )
            )

            chemin = generer_un_clip(
                video=video,
                clip=clip,
                index=index,
                dossier=dossier,
                transcription=transcription,
                format_video=format_video,
                sous_titres_actifs=sous_titres_actifs,
                emojis_actifs=emojis_actifs
            )

            resultats.append(
                chemin
            )

        progress(
            1.0,
            desc="Tous les clips sont prêts !"
        )

        statut = (
            f"Terminé ! {len(resultats)} clip(s) créé(s).\n"
            f"Durée de la vidéo : "
            f"{int(duree // 60)} min "
            f"{int(duree % 60)} sec.\n"
            f"Format : {format_video}.\n"
            f"Sous-titres : "
            f"{'Activés' if sous_titres_actifs else 'Désactivés'}.\n"
            f"Émojis : "
            f"{'Activés' if emojis_actifs else 'Désactivés'}."
        )

        return resultats, statut

    except gr.Error:
        raise

    except Exception as erreur:
        raise gr.Error(
            "Une erreur est survenue : "
            + str(erreur)[-1800:]
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

.bloc-options {
    border-radius: 16px;
}
"""

                video = gr.Video(
                    label="📁 Vidéo du podcast",
                    sources=["upload"],
                    type="filepath"
                )

                nombre_clips = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Nombre de clips"
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
                gr.Markdown("### ⚙️ Personnalisation")

                sous_titres_actifs = gr.Checkbox(
                    value=True,
                    label="💬 Activer les sous-titres"
                )

                emojis_actifs = gr.Checkbox(
                    value=True,
                    label="🚀 Activer les émojis business"
                )

        bouton = gr.Button(
            "🚀 GÉNÉRER MES CLIPS",
            variant="primary",
            size="lg"
        )

        statut = gr.Textbox(
            label="📊 Progression et résultat",
            interactive=False,
            lines=5
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
            outputs=[fichiers, statut]
        )

        gr.Markdown(
            """
            ---
            **ClipIA V7** · Génération automatique de clips
            """
        )

if __name__ == "__main__":
    application.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=CSS
)
                
