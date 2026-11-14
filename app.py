
import os
import cv2
import subprocess
import tempfile
import gradio as gr
import imageio_ffmpeg
from faster_whisper import WhisperModel

modele_whisper = None


def obtenir_modele():
    global modele_whisper

    if modele_whisper is None:
        modele_whisper = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8"
        )

    return modele_whisper


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


def analyser_visuellement(capture, fps, total):
    scores = []
    precedente = None
    intervalle = max(1, int(fps * 2))
    compteur = 0

    while compteur < total:
        capture.set(cv2.CAP_PROP_POS_FRAMES, compteur)
        ok, frame = capture.read()

        if not ok:
            break

        petite = cv2.resize(frame, (96, 54))
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

        scores.append((compteur / fps, score))
        precedente = gris
        compteur += intervalle

    return scores


def choisir_passages(
    segments,
    scores_visuels,
    duree,
    longueur,
    nombre
):
    candidats = []

    pas = 5
    dernier_debut = max(0, duree - longueur)

    debuts = list(range(0, int(dernier_debut) + 1, pas))

    if not debuts or debuts[-1] < dernier_debut:
        debuts.append(dernier_debut)

    mots_cles = [
        "incroyable", "attends", "pourquoi",
        "comment", "jamais", "vraiment",
        "énorme", "attention", "regarde",
        "sérieux", "wow", "haha",
        "bravo", "non", "oh"
    ]

    maximum_visuel = max(
        (score for temps, score in scores_visuels),
        default=0
    )

    for debut in debuts:
        fin = debut + longueur
        score = 0

        # Analyse des paroles
        for segment in segments:
            chevauchement = (
                segment["debut"] < fin
                and segment["fin"] > debut
            )

            if chevauchement:
                texte = segment["texte"].lower()

                score += len(texte.split())

                score += sum(
                    2 for mot in mots_cles
                    if mot in texte
                )

                score += texte.count("!") * 2
                score += texte.count("?")

        # Analyse des changements visuels
        if maximum_visuel > 0:
            activite = [
                valeur
                for temps, valeur in scores_visuels
                if debut <= temps <= fin
            ]

            if activite:
                score += (max(activite) / maximum_visuel) * 10

        if score > 0:
            candidats.append((debut, score))

    candidats.sort(key=lambda x: x[1], reverse=True)

    selection = []

    for debut, score in candidats:
        if all(
            abs(debut - ancien) >= longueur
            for ancien in selection
        ):
            selection.append(debut)

        if len(selection) >= nombre:
            break

    return selection


def analyser_video(video, nombre_clips, format_video):
    if video is None:
        raise gr.Error("Ajoute d'abord une vidéo.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dossier = tempfile.mkdtemp()

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise gr.Error("Impossible de lire la vidéo.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0 or total <= 0:
        capture.release()
        raise gr.Error("Vidéo invalide.")

    duree = total / fps
    nombre_clips = int(nombre_clips)
    longueur = min(20, duree)

    # Analyse visuelle de toute la vidéo
    scores_visuels = analyser_visuellement(
        capture,
        fps,
        total
    )

    # Extraction audio
    audio = os.path.join(dossier, "audio.wav")

    commande_audio = [
        ffmpeg, "-y",
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

    except subprocess.TimeoutExpired:
        capture.release()
        raise gr.Error(
            "L'extraction audio prend trop de temps."
        )

    except Exception:
        # Si l'analyse audio échoue,
        # on conserve l'analyse visuelle.
        segments = []

    # Combinaison des deux analyses
    intervalles = choisir_passages(
        segments,
        scores_visuels,
        duree,
        longueur,
        nombre_clips
    )

    capture.release()

    if not intervalles:
        intervalles = [0]

    clips = []

    for i, debut in enumerate(intervalles):
        sortie = os.path.join(
            dossier,
            f"clip_{i+1}.mp4"
        )

        commande = [
            ffmpeg, "-y",
            "-ss", str(debut),
            "-i", video,
            "-t", str(longueur)
        ]

        if format_video == "Vertical 9:16":
            filtre = (
                "[0:v]split=2[bg][fg];"
                "[bg]scale=360:640:force_original_aspect_ratio=increase,"
                "crop=360:640,boxblur=15:5[background];"
                "[fg]scale=360:640:force_original_aspect_ratio=decrease,"
                "setsar=1[foreground];"
                "[background][foreground]"
                "overlay=(W-w)/2:(H-h)/2[out]"
            )

            commande.extend([
                "-filter_complex", filtre,
                "-map", "[out]"
            ])

        else:
            commande.extend([
                "-map", "0:v:0"
            ])

        commande.extend([
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
            "-threads", "2",
            "-movflags", "+faststart",
            sortie
        ])

        try:
            subprocess.run(
                commande,
                capture_output=True,
                text=True,
                timeout=240,
                check=True
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

        if not os.path.exists(sortie) or os.path.getsize(sortie) == 0:
            raise gr.Error("Un clip n'a pas été créé.")

        clips.append(sortie)

    return clips


with gr.Blocks(title="ClipIA") as demo:

    gr.Markdown("# 🎬 ClipIA")
    gr.Markdown(
        "Analyse les paroles et les changements visuels "
        "pour trouver des passages intéressants."
    )

    video = gr.Video(label="Choisis ta vidéo")

    nombre = gr.Slider(
        minimum=1,
        maximum=5,
        value=3,
        step=1,
        label="Nombre de clips"
    )

    format_video = gr.Radio(
        choices=["Original", "Vertical 9:16"],
        value="Original",
        label="Format des clips"
    )

    bouton = gr.Button("🚀 Créer mes clips")

    resultat = gr.Gallery(label="Tes clips générés")

    bouton.click(
        analyser_video,
        inputs=[video, nombre, format_video],
        outputs=resultat
    )


demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
    )
                
            

