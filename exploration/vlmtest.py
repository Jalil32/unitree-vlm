import cv2
import time
import torch
from PIL import Image
import numpy as np
import argparse
from datetime import datetime
import os
import threading
import queue
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write as write_wav
import speech_recognition as sr
from gtts import gTTS
import pygame
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class ConversationalVisualAssistant:
    def __init__(self, camera_id=0, fps=5, save_output=False, output_dir="output"):
        """
        Initialize the conversational visual assistant.

        Args:
            camera_id (int): Camera device ID
            fps (int): Target frames per second for video processing
            save_output (bool): Whether to save output files
            output_dir (str): Directory to save output files
        """
        # Basic settings
        self.camera_id = camera_id
        self.fps = fps
        self.save_output = save_output
        self.output_dir = output_dir
        self.frame_delay = 1.0 / fps
        self.running = False
        self.speaking = False
        self.listening = False

        # Create output directory if needed
        if save_output and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Set up communication queues
        self.command_queue = queue.Queue()  # Commands from speech recognition
        self.response_queue = queue.Queue()  # Responses to be spoken
        self.current_frame = None  # Store the current camera frame

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize pygame for audio playback
        pygame.mixer.init()

        # Load models
        self.load_models()

    def load_models(self):
        """Load the required models for vision and speech processing."""
        print("Loading models...")

        # Load BLIP-2 model
        try:
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16
            ).to(self.device)
            print("BLIP-2 model loaded successfully")
        except Exception as e:
            print(f"Error loading BLIP-2 model: {e}")
            raise

        # Load YOLO model for object detection if available
        try:
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)
            self.has_yolo = True
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"YOLO model loading failed: {e}")
            print("Object detection will be disabled")
            self.has_yolo = False

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        print("Speech recognition initialized")

        print("All models loaded successfully")

    def start(self):
        """Start all processing threads and begin the assistant."""
        self.running = True

        # Start the camera thread
        self.camera_thread = threading.Thread(target=self.camera_processing_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        # Start the speech recognition thread
        self.speech_thread = threading.Thread(target=self.speech_recognition_loop)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        # Start the command processing thread
        self.command_thread = threading.Thread(target=self.command_processing_loop)
        self.command_thread.daemon = True
        self.command_thread.start()

        # Start the text-to-speech thread
        self.tts_thread = threading.Thread(target=self.text_to_speech_loop)
        self.tts_thread.daemon = True
        self.tts_thread.start()

        print("All threads started. Assistant is running.")
        print("Speak to the assistant. Say 'describe' to ask for a description of what's visible.")

        # Wait for a quit command
        try:
            while self.running:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False

        # Clean up
        cv2.destroyAllWindows()
        print("Assistant stopped.")

    def camera_processing_loop(self):
        """Main loop for camera processing."""
        # Open the camera
        cap = cv2.VideoCapture(self.camera_id)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            self.running = False
            return

        print(f"Camera opened successfully. Target FPS: {self.fps}")

        try:
            while self.running:
                # Capture frame
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Store the current frame for processing
                self.current_frame = frame.copy()

                # Display status on frame
                status_frame = frame.copy()

                # Add status indicators
                cv2.putText(status_frame, f"Listening: {'Yes' if self.listening else 'No'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.listening else (0, 0, 255), 2)
                cv2.putText(status_frame, f"Speaking: {'Yes' if self.speaking else 'No'}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.speaking else (0, 0, 255), 2)

                # Display help text
                cv2.putText(status_frame, "Say 'describe' to analyze the scene",
                            (10, status_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display the frame
                cv2.imshow('Conversational Visual Assistant', status_frame)

                # Control frame rate
                time.sleep(self.frame_delay)

        finally:
            cap.release()

    def process_frame(self, frame):
        """
        Process a single frame and generate a description.

        Args:
            frame: The image frame to process

        Returns:
            str: Caption describing the frame
        """
        # Convert the frame from BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for BLIP-2 processing
        img = Image.fromarray(image_rgb)

        # Process the image with YOLO if available
        objects_detected = []
        if self.has_yolo:
            # Get detections
            results = self.yolo_model(image_rgb)
            detections = results.pandas().xyxy[0]

            # Get object names with confidence > 0.5
            confident_detections = detections[detections['confidence'] >= 0.5]
            objects_detected = confident_detections['name'].tolist()

        # Create a prompt based on detected objects
        if objects_detected:
            detected_str = ", ".join(set(objects_detected))
            prompt = f"I can see {detected_str}. Please describe this scene in detail for a visually impaired person."
        else:
            prompt = ""

        # Process the image and generate description
        inputs = self.processor(img, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=100)

        # Decode the caption
        caption = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return caption

    def speech_recognition_loop(self):
        """Listen for voice commands continuously."""
        print("Speech recognition started. Listening for commands...")

        with sr.Microphone() as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source)

            while self.running:
                if self.speaking:
                    # Don't listen while speaking to avoid feedback
                    time.sleep(0.5)
                    continue

                try:
                    self.listening = True
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                    self.listening = False

                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"Recognized: {text}")

                        # Add to command queue
                        self.command_queue.put(text)
                    except sr.UnknownValueError:
                        # Speech wasn't understood
                        pass
                    except sr.RequestError as e:
                        print(f"Speech recognition service error: {e}")
                except Exception as e:
                    self.listening = False
                    print(f"Listening error: {e}")
                    time.sleep(1)

    def command_processing_loop(self):
        """Process voice commands from the queue."""
        while self.running:
            try:
                # Get a command from the queue
                command = self.command_queue.get(timeout=1)

                if "describe" in command or "what do you see" in command or "tell me what you see" in command:
                    print("Processing 'describe' command...")

                    # Get the current frame
                    if self.current_frame is not None:
                        # Speak a prompt
                        self.response_queue.put("Let me look at what's in front of me.")

                        # Process the frame
                        caption = self.process_frame(self.current_frame)

                        # Add to response queue
                        self.response_queue.put(caption)
                    else:
                        self.response_queue.put("I can't see anything right now.")

                elif "exit" in command or "quit" in command or "stop" in command:
                    self.response_queue.put("Stopping the assistant.")
                    time.sleep(3)  # Wait for speech to complete
                    self.running = False

                elif "hello" in command or "hi" in command:
                    self.response_queue.put("Hello! I'm your visual assistant. Ask me to describe what I see.")

                else:
                    self.response_queue.put("I heard you say " + command + ". You can ask me to describe what I see.")

                # Mark command as processed
                self.command_queue.task_done()

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Command processing error: {e}")

    def text_to_speech_loop(self):
        """Convert text responses to speech."""
        while self.running:
            try:
                # Get a response from the queue
                response = self.response_queue.get(timeout=1)

                if response:
                    print(f"Speaking: {response}")
                    self.speaking = True

                    # Generate speech
                    tts = gTTS(text=response, lang='en')
                    temp_file = "temp_speech.mp3"
                    tts.save(temp_file)

                    # Play the speech
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)

                    # Remove temporary file
                    try:
                        os.remove(temp_file)
                    except:
                        pass

                    self.speaking = False

                # Mark response as processed
                self.response_queue.task_done()

            except queue.Empty:
                pass
            except Exception as e:
                self.speaking = False
                print(f"Text-to-speech error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Conversational Visual Assistant")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--fps", type=int, default=5, help="Target frames per second (default: 5)")
    parser.add_argument("--save", action="store_true", help="Save output files")
    parser.add_argument("--output", type=str, default="output", help="Output directory (default: 'output')")
    args = parser.parse_args()

    # Create and start the assistant
    assistant = ConversationalVisualAssistant(
        camera_id=args.camera,
        fps=args.fps,
        save_output=args.save,
        output_dir=args.output
    )

    assistant.start()

if __name__ == "__main__":
    main()
