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
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import json
import random

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

        # Conversation history and context
        self.conversation_history = []
        self.last_description = ""
        self.scene_memory = {}  # Store information about the scene
        self.user_preferences = {}  # Store user preferences
        self.last_interaction_time = time.time()

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

        # Personality and conversation templates
        self.load_conversation_templates()

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

        # Load conversation model if available
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
            self.llm_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(self.device)
            self.has_llm = True
            print("Conversation model loaded successfully")
        except Exception as e:
            print(f"Conversation model loading failed: {e}")
            print("Using template-based responses instead")
            self.has_llm = False

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        print("Speech recognition initialized")

        print("All models loaded successfully")

    def load_conversation_templates(self):
        """Load templates for more natural conversation."""
        # Greetings
        self.greetings = [
            "Hello! I'm your visual assistant. How can I help you today?",
            "Hi there! I'm ready to assist you. What would you like to know about your surroundings?",
            "Good to see you! I'm here to be your eyes. What can I help you with?",
            "Hey! I'm your visual companion. Just let me know what you'd like me to describe."
        ]

        # Description intros
        self.description_intros = [
            "Let me look at what's in front of me...",
            "I'm analyzing the scene...",
            "Let me see what's here...",
            "Taking a look around...",
            "Examining what's in view..."
        ]

        # Follow-up questions
        self.follow_ups = [
            "Would you like me to focus on any particular part of the scene?",
            "Is there something specific you'd like me to describe in more detail?",
            "Would you like to know more about any particular object I mentioned?",
            "Is there anything else you'd like to know about what I see?"
        ]

        # Change notices
        self.change_notices = [
            "I notice something has changed in the scene. {change}",
            "Something's different now. {change}",
            "The scene has changed. {change}",
            "I'm seeing a change in the environment. {change}"
        ]

        # Continuity phrases
        self.continuity_phrases = [
            "As I mentioned before, {reference}...",
            "Just like before, {reference}...",
            "Still {reference}...",
            "As I can still see, {reference}..."
        ]

        # Idle conversation starters
        self.idle_conversation = [
            "It's been quiet for a moment. Would you like me to describe anything I see?",
            "I'm still here if you need me. Just ask if you want me to describe the scene.",
            "Let me know if you'd like me to tell you about anything in your surroundings.",
            "I'm ready whenever you are. Just say the word if you need a description."
        ]

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

        # Start the idle conversation thread
        self.idle_thread = threading.Thread(target=self.idle_conversation_loop)
        self.idle_thread.daemon = True
        self.idle_thread.start()

        # Welcome message
        welcome = random.choice(self.greetings)
        self.response_queue.put(welcome)

        print("All threads started. Assistant is running.")
        print("Speak to the assistant. Try asking 'what do you see?' or 'describe what's around me'")

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
        """Main loop for camera processing and scene change detection."""
        # Open the camera
        cap = cv2.VideoCapture(self.camera_id)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            self.running = False
            return

        print(f"Camera opened successfully. Target FPS: {self.fps}")

        # Variables for scene change detection
        last_objects = set()
        last_scene_check = time.time()
        scene_check_interval = 5  # Check for scene changes every 5 seconds

        try:
            while self.running:
                # Capture frame
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Store the current frame for processing
                self.current_frame = frame.copy()

                # Check for scene changes periodically
                current_time = time.time()
                if current_time - last_scene_check > scene_check_interval and not self.speaking:
                    # Detect objects in the current scene
                    current_objects = self.detect_objects_in_frame(frame)
                    current_objects_set = set(current_objects)

                    # Check for significant changes
                    if last_objects and current_objects_set:
                        new_objects = current_objects_set - last_objects
                        removed_objects = last_objects - current_objects_set

                        if (len(new_objects) >= 2 or len(removed_objects) >= 2) and not self.speaking:
                            # Significant change detected
                            change_message = ""
                            if new_objects:
                                change_message += f"I now see {', '.join(new_objects)}. "
                            if removed_objects:
                                change_message += f"I no longer see {', '.join(removed_objects)}."

                            change_notice = random.choice(self.change_notices).format(change=change_message)
                            self.response_queue.put(change_notice)

                    # Update the last objects
                    last_objects = current_objects_set
                    last_scene_check = current_time

                # Display status on frame
                status_frame = frame.copy()

                # Add status indicators
                cv2.putText(status_frame, f"Listening: {'Yes' if self.listening else 'No'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.listening else (0, 0, 255), 2)
                cv2.putText(status_frame, f"Speaking: {'Yes' if self.speaking else 'No'}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.speaking else (0, 0, 255), 2)

                # Display conversation history snippet
                if self.conversation_history:
                    last_exchange = self.conversation_history[-1]
                    y_pos = 100
                    cv2.putText(status_frame, f"You: {last_exchange['user'][:50]}",
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    response_lines = [last_exchange['assistant'][i:i+50] for i in range(0, min(len(last_exchange['assistant']), 150), 50)]
                    for line in response_lines:
                        y_pos += 25
                        cv2.putText(status_frame, f"Assistant: {line}",
                                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Display help text
                cv2.putText(status_frame, "Say 'describe' or ask a question about what you see",
                            (10, status_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display the frame
                cv2.imshow('Conversational Visual Assistant', status_frame)

                # Control frame rate
                time.sleep(self.frame_delay)

        finally:
            cap.release()

    def detect_objects_in_frame(self, frame):
        """Detect objects in a frame and return a list of object names."""
        if not self.has_yolo:
            return []

        try:
            # Convert the frame from BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get detections
            results = self.yolo_model(image_rgb)
            detections = results.pandas().xyxy[0]

            # Get object names with confidence > 0.5
            confident_detections = detections[detections['confidence'] >= 0.5]
            objects_detected = confident_detections['name'].tolist()

            return objects_detected
        except Exception as e:
            print(f"Object detection error: {e}")
            return []

    def process_frame(self, frame, focus=None):
        """
        Process a single frame and generate a description.

        Args:
            frame: The image frame to process
            focus: Optional focus area or object

        Returns:
            str: Caption describing the frame
        """
        # Convert the frame from BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for BLIP-2 processing
        img = Image.fromarray(image_rgb)

        # Process the image with YOLO if available
        objects_detected = self.detect_objects_in_frame(frame)
        self.scene_memory['objects'] = objects_detected

        # Create a prompt based on detected objects and conversation history
        prompt = self.create_vision_prompt(objects_detected, focus)

        # Process the image and generate description
        inputs = self.processor(img, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=100)

        # Decode the caption
        caption = self.processor.batch_decode(output, skip_special_tokens=True)[0]

        # Store the description
        self.last_description = caption
        self.scene_memory['last_description'] = caption

        return caption

    def create_vision_prompt(self, objects_detected, focus=None):
        """Create a prompt for the vision model based on context."""
        # Start with a basic prompt
        prompt = "Please describe what you see in this image"

        # Add focus if specified
        if focus:
            prompt += f" with special attention to the {focus}"

        # Add detected objects context
        if objects_detected:
            detected_str = ", ".join(set(objects_detected))
            prompt += f". I can see {detected_str}"

        # Add conversational context
        if len(self.conversation_history) > 0:
            last_user_query = self.conversation_history[-1]['user']
            prompt += f". The user asked: '{last_user_query}'"

        # Complete the prompt
        prompt += ". Provide a friendly and detailed description as if talking to a friend."

        return prompt

    def generate_conversation_response(self, user_input, scene_context):
        """Generate a conversational response based on user input and scene context."""
        # Use LLM if available
        if self.has_llm:
            # Create a prompt for the LLM
            conversation_context = "You are a helpful visual assistant having a conversation with a user."
            conversation_context += f"\nObjects in the scene: {', '.join(scene_context.get('objects', []))}"
            conversation_context += f"\nLast description: {scene_context.get('last_description', '')}"

            # Add previous conversation turns
            conversation_context += "\n\nConversation history:"
            for turn in self.conversation_history[-3:]:  # Include last 3 turns
                conversation_context += f"\nUser: {turn['user']}\nAssistant: {turn['assistant']}"

            # Add current user input
            conversation_context += f"\nUser: {user_input}\nAssistant:"

            # Generate response with the LLM
            input_ids = self.llm_tokenizer(conversation_context, return_tensors="pt").input_ids.to(self.device)

            with torch.no_grad():
                output = self.llm_model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 100,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )

            response = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract just the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()

            return response

        # Fall back to template-based responses
        else:
            # Extract keywords from user input
            keywords = user_input.lower().split()

            # Check for description request
            if any(word in keywords for word in ["describe", "see", "look", "what's", "whats", "around"]):
                return random.choice(self.description_intros)

            # Check for greeting
            elif any(word in keywords for word in ["hello", "hi", "hey", "greetings"]):
                return random.choice(self.greetings)

            # Check for follow-up on previous description
            elif any(word in keywords for word in ["more", "detail", "explain", "about"]) and self.last_description:
                object_focus = None
                for obj in scene_context.get('objects', []):
                    if obj.lower() in user_input.lower():
                        object_focus = obj
                        break

                if object_focus:
                    return f"Let me tell you more about the {object_focus}..."
                else:
                    return "Let me provide more details about what I see..."

            # Generic response
            else:
                return "I'm not sure how to respond to that. Would you like me to describe what I see?"

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

                        # Update last interaction time
                        self.last_interaction_time = time.time()

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

                # Process command and generate response
                if "describe" in command or "what do you see" in command or "tell me what you see" in command:
                    # Get response to acknowledge the command
                    intro_response = self.generate_conversation_response(command, self.scene_memory)
                    self.response_queue.put(intro_response)

                    # Get the current frame
                    if self.current_frame is not None:
                        # Process the frame
                        caption = self.process_frame(self.current_frame)

                        # Add the option for a follow-up question
                        if random.random() < 0.7:  # 70% chance to add a follow-up
                            caption += " " + random.choice(self.follow_ups)

                        # Add to response queue
                        self.response_queue.put(caption)

                        # Add to conversation history
                        self.conversation_history.append({
                            'user': command,
                            'assistant': caption
                        })
                    else:
                        response = "I can't see anything right now."
                        self.response_queue.put(response)
                        self.conversation_history.append({
                            'user': command,
                            'assistant': response
                        })

                elif "exit" in command or "quit" in command or "stop" in command:
                    response = "Stopping the assistant. It was nice talking with you!"
                    self.response_queue.put(response)
                    self.conversation_history.append({
                        'user': command,
                        'assistant': response
                    })
                    time.sleep(3)  # Wait for speech to complete
                    self.running = False

                else:
                    # Generate a conversational response
                    response = self.generate_conversation_response(command, self.scene_memory)
                    self.response_queue.put(response)

                    # If it seems like a question about the scene but not a direct "describe" command
                    if any(word in command for word in ["what", "where", "how many", "is there", "do you see"]):
                        # Check if we should process the frame again
                        if self.current_frame is not None:
                            # Find focus object if any
                            focus = None
                            for obj in self.scene_memory.get('objects', []):
                                if obj.lower() in command.lower():
                                    focus = obj
                                    break

                            # Process the frame with possible focus
                            caption = self.process_frame(self.current_frame, focus)
                            self.response_queue.put(caption)

                            # Update conversation history
                            self.conversation_history.append({
                                'user': command,
                                'assistant': f"{response} {caption}"
                            })
                        else:
                            self.conversation_history.append({
                                'user': command,
                                'assistant': response
                            })
                    else:
                        self.conversation_history.append({
                            'user': command,
                            'assistant': response
                        })

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

    def idle_conversation_loop(self):
        """Periodically check if the system has been idle and initiate conversation."""
        idle_threshold = 60  # 60 seconds of no interaction

        while self.running:
            current_time = time.time()
            time_since_last_interaction = current_time - self.last_interaction_time

            # If idle for too long and not currently speaking/listening
            if (time_since_last_interaction > idle_threshold and
                not self.speaking and not self.listening and
                random.random() < 0.3):  # 30% chance to initiate conversation

                # Reset the timer
                self.last_interaction_time = current_time

                # Choose an idle message
                idle_message = random.choice(self.idle_conversation)
                self.response_queue.put(idle_message)

            # Sleep to avoid busy waiting
            time.sleep(10)  # Check every 10 seconds

def main():
    parser = argparse.ArgumentParser(description="Enhanced Conversational Visual Assistant")
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
