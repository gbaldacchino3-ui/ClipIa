import os
import cv2
import subprocess
import tempfile
import gradio as gr
import imageio_ffmpeg
from faster_whisper import WhisperModel

# =====================================
# CONFIGURATION
# =====================================

modele_whisper = None

DUREE_MAX_CLIP = 20
INTERVALLE_VISUEL = 4


# =====================================
# WHISPER
# =====================================

def obtenir_modele():
    global modele_whisper

    if modele_whisper is None:
        modele_whisper = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
            num_workers=1
        )

    return modele_whisper


def analyser_paroles(audio):
    modele = obtenir_modele()

    segments, infos = modele.transcribe(
        audio,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False
    )

    resultat = []

    for segment in segments:
        texte = segment.text.strip()

        if texte:
            resultat.append({
                "debut": segment.start,
                "fin": segment.end,
                "texte": texte
            })

    return resultat


# =====================================
# ANALYSE VISUELLE ALLEGEE
# =====================================

def analyser_visuellement(capture, fps, total):

    scores = []
    precedente = None

    intervalle = max(
        1,
        int(fps * INTERVALLE_VISUEL)
    )

    compteur = 0

    while compteur < total:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            compteur
        )

        ok, frame = capture.read()

        if not ok:
            break

        petite = cv2.resize(
            frame,
            (64, 36)
        )

        gris = cv2.cvtColor(
            petite,
            cv2.COLOR_BGR2GRAY
        )

        score = 0

        if precedente is not None:
            score = cv2.absdiff(
                gris,
                precedente
            ).mean()

        scores.append((
            compteur / fps,
            score
        ))

        precedente = gris
        compteur += intervalle

    return scores


# =====================================
# SCORING DES PAROLES
# =====================================

def score_paroles(segments, debut, fin):

    score = 0

    mots_reaction = [
        "wow", "haha", "ahah", "mdr",
        "incroyable", "attends", "pourquoi",
        "comment", "jamais", "vraiment",
        "énorme", "attention", "regarde",
        "sérieux", "bravo", "non", "oh",
        "quoi", "impossible", "mince",
        "oups", "surprise", "excellent"
    ]

    for segment in segments:

        if (
            segment["debut"] < fin
            and segment["fin"] > debut
        ):

            texte = segment["texte"].lower()

            score += len(texte.split())

            for mot in mots_reaction:
                if mot in texte:
                    score += 4

            score += texte.count("!") * 3
            score += texte.count("?") * 2

    return score


# =====================================
# SCORING VISUEL
# =====================================

def score_visuel(scores_visuels, debut, fin):

    valeurs = [
        valeur
        for temps, valeur in scores_visuels
        if debut <= temps <= fin
    ]

    if not valeurs:
        return 0

    return (
        sum(valeurs) / len(valeurs)
        + max(valeurs) * 0.5
    )


# =====================================
# CHOIX DES MEILLEURS PASSAGES
# =====================================

def choisir_passages(
    segments,
    scores_visuels,
    duree,
    longueur,
    nombre
):

    dernier_debut = max(
        0,
        duree - longueur
    )

    pas = 2

    debuts = list(
        range(
            0,
            int(dernier_debut) + 1,
            pas
        )
    )

    if (
        not debuts
        or debuts[-1] < dernier_debut
    ):
        debuts.append(dernier_debut)

    maximum_visuel = max(
        (
            score
            for temps, score in scores_visuels
        ),
        default=0
    )

    candidats = []

    for debut in debuts:

        fin = min(
            duree,
            debut + longueur
        )

        paroles = score_paroles(
            segments,
            debut,
            fin
        )

        visuel = score_visuel(
            scores_visuels,
            debut,
            fin
        )

        if maximum_visuel > 0:
            visuel = (
                visuel / maximum_visuel
            ) * 20

        score = paroles + visuel

        candidats.append((
            debut,
            score
        ))

    candidats.sort(
        key=lambda x: x[1],
        reverse=True
    )

    selection = []

    # Sélection prioritaire des passages distincts
    for debut, score in candidats:

        if all(
            abs(debut - ancien) >= 1
            for ancien in selection
        ):
            selection.append(debut)

        if len(selection) >= nombre:
            break

    # Compléter si nécessaire
    for debut, score in candidats:

        if len(selection) >= nombre:
            break

        if debut not in selection:
            selection.append(debut)

    return sorted(selection[:nombre])


# =====================================
# CREATION RAPIDE DES CLIPS
# =====================================

def creer_clip(
    ffmpeg,
    video,
    sortie,
    debut,
    longueur,
    format_video
):

    base = [
        ffmpeg,
        "-y",
        "-ss",
        str(debut),
        "-i",
        video,
        "-t",
        str(longueur)
    ]

    # FORMAT ORIGINAL : COPIE SANS REENCODAGE
    if format_video == "Original":

        commande = base + [
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            sortie
        ]

        try:
            subprocess.run(
                commande,
                capture_output=True,
                text=True,
                timeout=120,
                check=True
            )

            if (
                os.path.exists(sortie)
                and os.path.getsize(sortie) > 0
            ):
                return

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired
        ):
            pass

        # Si la copie échoue, réencoder normalement
        if os.path.exists(sortie):
            os.remove(sortie)

        commande = base + [
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-threads", "2",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
            "-movflags", "+faststart",
            sortie
        ]

    # FORMAT VERTICAL
    else:

        filtre = (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=270:480:"
            "force_original_aspect_ratio=increase,"
            "crop=270:480,boxblur=8:2[background];"
            "[fg]scale=270:480:"
            "force_original_aspect_ratio=decrease,"
            "setsar=1[foreground];"
            "[background][foreground]"
            "overlay=(W-w)/2:(H-h)/2[out]"
        )

        commande = base + [
            "-filter_complex", filtre,
            "-map", "[out]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "30",
            "-threads", "2",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "64k",
            "-movflags", "+faststart",
            sortie
        ]

    subprocess.run(
        commande,
        capture_output=True,
        text=True,
        timeout=240,
        check=True
    )

    if (
        not os.path.exists(sortie)
        or os.path.getsize(sortie) == 0
    ):
        raise gr.Error(
            "Un clip n'a pas été créé."
        )


# =====================================
# ANALYSE PRINCIPALE
# =====================================

def analyser_video(
    video,
    nombre_clips,
    format_video
):

    if video is None:
        raise gr.Error(
            "Ajoute d'abord une vidéo."
        )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dossier = tempfile.mkdtemp()

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise gr.Error(
            "Impossible de lire la vidéo."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    total = capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    if fps <= 0 or total <= 0:
        capture.release()

        raise gr.Error(
            "Vidéo invalide."
        )

    duree = total / fps

    nombre_clips = int(nombre_clips)

    longueur = min(
        DUREE_MAX_CLIP,
        duree
    )

    # ANALYSE VISUELLE
    try:

        scores_visuels = analyser_visuellement(
            capture,
            fps,
            total
        )

    except Exception as e:

        capture.release()

        raise gr.Error(
            "Erreur visuelle : " + str(e)
        )

    capture.release()

    # EXTRACTION AUDIO
    audio = os.path.join(
        dossier,
        "audio.wav"
    )

    commande_audio = [
        ffmpeg,
        "-y",
        "-i", video,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        audio
    ]

    segments = []

    try:

        subprocess.run(
            commande_audio,
            capture_output=True,
            text=True,
            timeout=90,
            check=True
        )

        segments = analyser_paroles(audio)

    except Exception:
        # Continuer avec l'analyse visuelle
        segments = []

    # SELECTION DES PASSAGES
    intervalles = choisir_passages(
        segments,
        scores_visuels,
        duree,
        longueur,
        nombre_clips
    )

    if not intervalles:
        intervalles = [0]

    # CREATION DES CLIPS
    clips = []

    for i, debut in enumerate(intervalles):

        sortie = os.path.join(
            dossier,
            f"clip_{i+1}.mp4"
        )

        try:

            creer_clip(
                ffmpeg,
                video,
                sortie,
                debut,
                longueur,
                format_video
            )

        except subprocess.TimeoutExpired:

            raise gr.Error(
                "Traitement trop long. "
                "Essaie une vidéo plus courte."
            )

        except subprocess.CalledProcessError as e:

            raise gr.Error(
                "Erreur FFmpeg : " + e.stderr[-1000:]
            )

        clips.append(sortie)

    return clips


# =====================================
# INTERFACE
# =====================================

with gr.Blocks(
    title="ClipIA V3",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🎬 ClipIA V3

        ### Création intelligente de clips

        Analyse les paroles et les changements visuels
        pour trouver des passages intéressants.
        """
    )

    video = gr.Video(
        label="Choisis ta vidéo"
    )

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
        value="Original",
        label="Format des clips"
    )

    bouton = gr.Button(
        "🚀 Créer mes clips",
        variant="primary"
    )

    resultat = gr.Gallery(
        label="Tes clips générés",
        columns=2
    )

    bouton.click(
        analyser_video,
        inputs=[
            video,
            nombre,
            format_video
        ],
        outputs=resultat
    )

    gr.Markdown(
        """
        ---
        **ClipIA V3** · Performance optimisée
        """
    )


demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(
        os.environ.get("PORT", 7860)
    )
    )
        
