#VoiceBot UI with Gradio
import os
from dotenv import load_dotenv
import gradio as gr
import logging

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# Verify API keys are loaded
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

#load_dotenv()

system_prompt="""You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_inputs(image, audio):
    try:
        # Handle audio input
        if audio is None:
            speech_to_text_output = "No audio description provided."
        else:
            try:
                speech_to_text_output = transcribe_with_groq("whisper-1", audio, GROQ_API_KEY)
            except Exception as e:
                logging.error(f"Error in audio transcription: {str(e)}")
                speech_to_text_output = "Error transcribing audio. Please try again."

        # Handle image input
        if image is None:
            doctor_response = "No image provided for analysis"
        else:
            try:
                encoded_image = encode_image(image)
                doctor_response = analyze_image_with_query(
                    query=system_prompt + " " + speech_to_text_output,
                    model="llama-3.2-90b-vision-preview",
                    encoded_image=encoded_image
                )
            except Exception as e:
                logging.error(f"Error in image analysis: {str(e)}")
                doctor_response = "Error analyzing image. Please try again."

        # Generate voice response
        voice_of_doctor = text_to_speech_with_elevenlabs(
            input_text=doctor_response,
            output_filepath="final.mp3"
        )

        return speech_to_text_output, doctor_response, voice_of_doctor

    except Exception as e:
        logging.error(f"Error in process_inputs: {str(e)}")
        return "An error occurred. Please try again.", "An error occurred. Please try again.", "final.mp3"


# Create Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Image(type="filepath", label="Upload Medical Image"),
        gr.Audio(type="filepath", label="Record or Upload Audio Description")
    ],
    outputs=[
        gr.Textbox(label="Transcribed Audio"),
        gr.Textbox(label="Doctor's Analysis"),
        gr.Audio(label="Response Audio", type="filepath")
    ],
    title="AI Medical Assistant",
    description="Upload a medical image and provide an audio description for analysis."
)

if __name__ == "__main__":
    iface.launch(debug=True)