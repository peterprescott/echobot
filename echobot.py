import os
import subprocess
import vosk
import json
import pyaudio
import ollama

# Initialize Vosk model and recognizer
model = vosk.Model("vosk-model/vosk-model-small-en-us-0.15")
recognizer = vosk.KaldiRecognizer(model, 16000)

# Set up the microphone for speech input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4000)
stream.start_stream()

def speak(text):
    """Uses espeak to convert text to speech."""
    subprocess.run(['espeak', text])

def chat_with_bot(text):
    """Send text to Ollama LLM for a response."""
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": text}])
    return response.message.content

def main():
    print("Start speaking...")
    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            input_text = result.get("text", "")
            if input_text:
                print(f"You said: {input_text}")
                bot_response = chat_with_bot(input_text)
                print(f"Bot: {bot_response}")
                speak(bot_response)

if __name__ == "__main__":
    main()
