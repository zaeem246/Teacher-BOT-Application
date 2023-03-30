import pyttsx3
import openai
import sounddevice as sd
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_file
import threading
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

openai.api_key = "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"
model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")


client = MongoClient("mongodb://zaeem246:OQaZZuVV5aCev6j1@ac-ivzommt-shard-00-00.yn2mctg.mongodb.net:27017,ac-ivzommt-shard-00-01.yn2mctg.mongodb.net:27017,ac-ivzommt-shard-00-02.yn2mctg.mongodb.net:27017/TeacherBot?ssl=true&replicaSet=atlas-te7kz1-shard-0&authSource=admin&retryWrites=true&w=majority")
db = client["TeacherBot"]
collection = db["mycollection"]


def get_similar_answer(transcribed_text):
    # Retrieve all documents from the database
    documents = collection.find()
    max_similarity = 0
    best_match = None
    transcribed_text_embedding = model.encode([transcribed_text])[0]
    for doc in documents:
        # Compute similarity between transcribed text and document transcription
        doc_embedding = model.encode([doc["transcription"]])[0]
        similarity = util.cos_sim(transcribed_text_embedding, doc_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = doc["answer"]
    if max_similarity >= 0.9:
        return best_match
    return None


def run_speech(answer):
    engine = pyttsx3.init()
    female_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    engine.setProperty('voice', female_voice_id, )
    engine.setProperty('rate', 150)
    engine.say(answer)
    engine.runAndWait()
    

@app.route("/teacher.gif")
def get_image():
    return send_file("teacher.gif")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/transcribe_audio", methods=["POST"])
def transcribe_audio():
    duration_str = request.form.get("duration")
    if not duration_str:
        return render_template("index.html", error="Duration cannot be empty")
    try:
        duration = float(duration_str)
    except ValueError:
        return jsonify({"error": "Invalid duration value"})
    audio_file_path = "audio_file.wav"
    try:
        # Set the sampling rate
        fs = 44100
        # Set the number of channels
        sd.default.channels = 1
        # Start recording audio for the specified duration
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        # Write the recorded audio to a file
        sf.write(audio_file_path, myrecording, fs)
        # Open the audio file and transcribe it using the OpenAI API
        with open(audio_file_path, "rb") as audio_file:
            transcribed_text = openai.Audio.transcribe("whisper-1", audio_file)
        # Use OpenAI API to answer the question
        prompt = f"What is {transcribed_text}?"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        answer = response.choices[0].text

        transcribed_text_str = str(transcribed_text)
        # Check if a similar question has been asked before and retrieve the corresponding answer from the database
        similar_answer = get_similar_answer(transcribed_text_str)
        if similar_answer is not None:
            # Use threading to run the speech and HTML rendering in separate threads
            speech_thread = threading.Thread(target=run_speech, args=(similar_answer,))
            speech_thread.start()
            return render_template(
                "index.html", transcription=transcribed_text, answer=similar_answer
            )
        else:
            # Use OpenAI API to answer the question
            prompt = f"What is {transcribed_text}?"
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            answer = response.choices[0].text
            answer_doc = {"transcription": transcribed_text_str, "answer": answer}
            collection.insert_one(answer_doc)
            # Use threading to run the speech and HTML rendering in separate threads
            speech_thread = threading.Thread(target=run_speech, args=(answer,))
            speech_thread.start()
            # Now passing the transcription result and spoken answer as variables
            return render_template(
                "index.html", transcription=transcribed_text, answer=answer
            )

    except Exception as e:
        return render_template("index.html", error=f"Error: {e}")
    
if __name__ == "__main__":
    app.debug= True
    app.run(host = '0.0.0.0', port = 5000)

    