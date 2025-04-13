# AI Doctor with Vision and Voice

AI Doctor with Vision and Voice is an interactive, AI-powered web application that provides virtual medical diagnostics by analyzing your voice and a scalp image. It leverages cutting-edge APIs and deep learning models to convert speech to text, analyze images, generate a medical diagnosis, and produce natural-sounding audio responses.

## Features

- **Speech-to-Text Conversion:** Uses the Groq API to transcribe patient audio.
- **Image Analysis:** Processes scalp images with a pre-trained deep learning model (ResNet50) using PyTorch and OpenCV.
- **Medical Diagnosis:** Generates an expert medical response via OpenAI's GPT-4.
- **Text-to-Speech Conversion:** Converts the AI-generated response into natural-sounding audio using ElevenLabs API.
- **Interactive Web Interface:** Built with Gradio for an intuitive user experience.

## Project Structure

- `gradio_app.py`: Main application integrating all components using Gradio.
- `voice_of_the_patient.py`: Manages audio recording and speech-to-text transcription.
- `voice_of_the_doctor.py`: Converts text responses into speech.
- `image_analysis.py`: Processes and analyzes scalp images.
- `doctor_response.py`: Generates the AI doctor’s response using OpenAI's API.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AI_Doctor.git
   cd AI_Doctor
