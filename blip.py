import cv2
import time
import torch
from PIL import Image
import numpy as np

def access_camera_with_captioning(camera_id=0, fps=7):
    """
    Access the camera at a specified frame rate and generate captions for each frame using BLIP-2.
    
    Args:
        camera_id (int): Camera device ID (default: 0 for primary camera)
        fps (int): Target frames per second (default: 7)
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load BLIP-2 model and processor
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16
        ).to(device)
        print("BLIP-2 model loaded successfully")
    except ImportError:
        print("Error: Please install the transformers library:")
        print("pip install transformers")
        return
    
    # Calculate the delay between frames to achieve target FPS
    frame_delay = 1.0 / fps
    
    # Open the camera
    cap = cv2.VideoCapture(camera_id)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print(f"Camera opened successfully. Target FPS: {fps}")
    
    # Set up the prompt for image captioning
    prompt = "Describe what you are seeing. Answer:"
    
    try:
        while True:
            # Record the start time of this frame
            start_time = time.time()
            
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Check if frame was captured successfully
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for BLIP-2 processing
            img = Image.fromarray(image_rgb)
            
            # Process the image and generate description
            inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output = model.generate(**inputs)
            
            caption = processor.batch_decode(output, skip_special_tokens=True)[0]
            print("BLIP-2 Caption:", caption)
            
            # Display the resulting frame
            # Add caption to the frame
            cv2.putText(frame, "Caption: " + caption[:50], (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if len(caption) > 50:
                cv2.putText(frame, caption[50:100], (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Camera Feed with BLIP-2 Captioning', frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Calculate how long to wait to maintain target FPS
            processing_time = time.time() - start_time
            wait_time = max(0, frame_delay - processing_time)
            
            # Print actual FPS (will likely be lower due to BLIP-2 processing)
            actual_fps = 1.0 / (processing_time + wait_time if wait_time > 0 else processing_time)
            print(f"Actual FPS: {actual_fps:.2f}")
            
            if wait_time > 0:
                time.sleep(wait_time)
            
    finally:
        # Release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run the camera with captioning
    access_camera_with_captioning()