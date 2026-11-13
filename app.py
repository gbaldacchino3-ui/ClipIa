
import gradio as gr

def analyser_video(video):
    if video is None:
        return "Ajoute d'abord une vidéo."

    return "Vidéo reçue ! ClipIA est prêt pour la prochaine étape."

with gr.Blocks(title="ClipIA") as demo:
    gr.Markdown("# 🎬 ClipIA")
    gr.Markdown("Transforme tes vidéos en clips.")

    video = gr.Video(label="Choisis ta vidéo")
    bouton = gr.Button("Analyser ma vidéo")
    resultat = gr.Textbox(label="Résultat")

    bouton.click(
        fn=analyser_video,
        inputs=video,
        outputs=resultat
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)

