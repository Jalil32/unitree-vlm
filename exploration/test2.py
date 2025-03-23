"""
Voice-Enabled Llama Chatbot (Improved for Conversational Responses)
----------------------------
This script creates a conversational assistant using:
- Llama-1B for text generation
- SpeechRecognition for audio input
- pyttsx3 for text-to-speech output
"""

import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import speech_recognition as sr
import pyttsx3
import threading
import time

class VoiceChatbot:
    def __init__(self):
        self.setup_speech_engine()
        self.setup_recognizer()
        print("Loading Llama-1B model... (this may take a moment)")
        self.setup_llama_model()
        print("Voice assistant ready! Speak into your microphone.")

    def setup_speech_engine(self):
        """Initialize the text-to-speech engine"""
        self.engine = pyttsx3.init()
        # Adjust voice properties if needed
        self.engine.setProperty('rate', 150)  # Speed of speech

        # Optional: Set a specific voice
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)  # Often female voice

    def setup_recognizer(self):
        """Initialize the speech recognition"""
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Adjust based on your mic sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            print("Calibrating for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)

    def setup_llama_model(self):
        """Load the Llama-1B model"""
        # Using huggingface's implementation of Llama
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small 1B version of Llama

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"  # Will use GPU if available
        )

        # Initialize conversation with a strong, clear system prompt
        self.system_prompt = (
            "You are a helpful AI assistant named Llama. "
            "You will respond directly to user questions in a conversational way. "
            "When asked to tell a story or explain something, provide a complete answer. "
            "Keep responses concise but complete. "
            "Don't continue the user's input - answer their question or respond to their request."
        )

        # Create an empty conversation history
        self.messages = []

    def listen(self):
        """Listen to microphone input and convert to text"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def format_prompt(self, user_input):
        """Format the prompt using the chat template for better responses"""
        # Use the model's chat template to properly format the conversation
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add previous conversation context (limited to last 5 exchanges to manage context length)
        for msg in self.messages[-10:]:
            messages.append(msg)

        # Add the current user input
        messages.append({"role": "user", "content": user_input})

        # Use the model's chat template to format the conversation
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_response(self, user_input):
        """Generate a response using the Llama model with improved prompt formatting"""
        if not user_input:
            return "I didn't catch that. Could you please repeat?"

        # Format the prompt using the chat template
        prompt = self.format_prompt(user_input)

        # Add user message to conversation history
        self.messages.append({"role": "user", "content": user_input})

        # Generate response
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=150,  # Limit response length
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,  # Discourage repetition
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the newly generated tokens (the model's response)
        new_tokens = generated_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Clean up response
        response = response.strip()
        # Remove any prefixes that might indicate continued conversation like "User:" or "Human:"
        for prefix in ["User:", "Human:", "Person:", "<human>:", "<user>:"]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        # Add assistant's response to conversation history
        self.messages.append({"role": "assistant", "content": response})

        print(f"Assistant: {response}")
        return response

    def speak(self, text):
        """Convert text to speech"""
        if not text:
            return

        self.engine.say(text)
        self.engine.runAndWait()

    def run(self):
        """Main loop for the chatbot"""
        greeting = "Hello! I'm your voice assistant powered by Llama. How can I help you today?"
        print(f"Assistant: {greeting}")
        self.speak(greeting)

        while True:
            user_input = self.listen()

            # Check for exit command
            if user_input and any(phrase in user_input.lower() for phrase in ["exit", "quit", "goodbye", "bye"]):
                farewell = "Goodbye! Have a nice day."
                print(f"Assistant: {farewell}")
                self.speak(farewell)
                break

            response = self.generate_response(user_input)
            self.speak(response)


if __name__ == "__main__":
    # Create and run the chatbot
    chatbot = VoiceChatbot()
    chatbot.run()
