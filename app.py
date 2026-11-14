
import os
import cv2
import subprocess
import tempfile
import gradio as gr
import imageio_ffmpeg
from faster_whisper import WhisperModel

# =====================================
# CONFIGURATION CLIPIA
# =====================================

modele_whisper = None

DUREE_MAX_CLIP = 20
PAS_ANALYSE = 2


# =====================================
# MODELE WHISPER
# =====================================

def obtenir_modele():
    global modele_whisper

    if modele_whisper is None:
        modele_whisper = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8"
        )

    return modele_whisper


# =====================================
# ANALYSE DES PAROLES
# =====================================

def analyser_paroles(audio):
    modele = obtenir_modele()

    segments, infos = modele.transcribe(
        audio,
        beam_size=1,
        vad_filter=True
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
# ANALYSE VISUELLE
# =====================================

def analyser_visuellement(capture, fps, total):

    scores = []
    precedente = None

    intervalle = max(1, int(fps * PAS_ANALYSE))
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
            (96, 54)
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
# DETECTION DES MOMENTS INTERESSANTS
# =====================================

def score_paroles(segments, debut, fin):

    score = 0

    mots_reaction = [
        "wow",
        "haha",
        "ahah",
        "mdr",
        "incroyable",
        "attends",
        "pourquoi",
        "comment",
        "jamais",
        "vraiment",
        "énorme",
        "attention",
        "regarde",
        "sérieux",
        "bravo",
        "non",
        "oh",
        "quoi",
        "impossible",
        "mince",
        "oups",
        "surprise",
        "excellent"
    ]

    for segment in segments:

        if (
            segment["debut"] < fin
            and segment["fin"] > debut
        ):

            texte = segment["texte"].lower()

            mots = texte.split()

            score += len(mots)

            for mot in mots_reaction:
                if mot in texte:
                    score += 4

            score += texte.count("!") * 3
            score += texte.count("?") * 2

    return score


def score_visuel(scores_visuels, debut, fin):

    valeurs = [
        valeur
        for temps, valeur in scores_visuels
        if debut <= temps <= fin
    ]

    if not valeurs:
        return 0

    moyenne = sum(valeurs) / len(valeurs)
    maximum = max(valeurs)

    return moyenne + maximum * 0.5


# =====================================
# SELECTION DES CLIPS
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

    pas = min(
        PAS_ANALYSE,
        max(0.5, dernier_debut / 10)
    )

    debuts = []

    temps = 0

    while temps < dernier_debut:
        debuts.append(temps)
        temps += pas

    debuts.append(dernier_debut)

    maximum_visuel = max(
        (score for temps, score in scores_visuels),
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

        # Combinaison paroles et activité visuelle
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

    # Priorité aux passages différents
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

    # Si la vidéo ne permet pas assez de passages,
    # utiliser des positions régulièrement espacées.
    if len(selection) < nombre and dernier_debut > 0:

        for i in range(nombre):

            position = (
                dernier_debut * i / max(1, nombre - 1)
            )

            if all(
                abs(position - ancien) >= 0.5
                for ancien in selection
            ):
                selection.append(position)

            if len(selection) >= nombre:
                break

    return sorted(selection[:nombre])


# =====================================
# CREATION DES CLIPS
# =====================================

def creer_clip(
    ffmpeg,
    video,
    sortie,
    debut,
    longueur,
    format_video
):

    commande = [
        ffmpeg,
        "-y",
        "-ss",
        str(debut),
        "-i",
        video,
        "-t",
        str(longueur)
    ]

    if format_video == "Vertical 9:16":

        filtre = (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=360:640:"
            "force_original_aspect_ratio=increase,"
            "crop=360:640,boxblur=15:5[background];"
            "[fg]scale=360:640:"
            "force_original_aspect_ratio=decrease,"
            "setsar=1[foreground];"
            "[background][foreground]"
            "overlay=(W-w)/2:(H-h)/2[out]"
        )

        commande.extend([
            "-filter_complex",
            filtre,
            "-map",
            "[out]"
        ])

    else:

        commande.extend([
            "-map",
            "0:v:0"
        ])

    commande.extend([
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        sortie
    ])

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
            "Un clip n'a pas été créé correctement."
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

    # 1. Analyse visuelle
    try:

        scores_visuels = analyser_visuellement(
            capture,
            fps,
            total
        )

    except Exception as e:

        capture.release()

        raise gr.Error(
            "Erreur analyse visuelle : " + str(e)
        )

    capture.release()

    # 2. Extraction audio
    audio = os.path.join(
        dossier,
        "audio.wav"
    )

    commande_audio = [
        ffmpeg,
        "-y",
        "-i",
        video,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
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

    except subprocess.TimeoutExpired:

        raise gr.Error(
            "L'extraction audio prend trop de temps."
        )

    except Exception:

        # Continuer avec l'analyse visuelle
        segments = []

    # 3. Choix des meilleurs passages
    intervalles = choisir_passages(
        segments,
        scores_visuels,
        duree,
        longueur,
        nombre_clips
    )

    if not intervalles:
        intervalles = [0]

    # 4. Création des clips
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
                "Le traitement est trop long. "
                "Essaie une vidéo plus courte."
            )

        except subprocess.CalledProcessError as e:

            raise gr.Error(
                "Erreur FFmpeg : " + e.stderr[-1000:]
            )

        clips.append(sortie)

    return clips


# =====================================
# INTERFACE CLIPIA
# =====================================

with gr.Blocks(
    title="ClipIA",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🎬 ClipIA V2

        ### Ton assistant intelligent de création de clips

        Analyse les paroles et les changements visuels
        pour repérer des passages potentiellement intéressants.
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
        **ClipIA V2** · Analyse audio et visuelle
        """
    )


demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(
        os.environ.get("PORT", 7860)
    )
)
