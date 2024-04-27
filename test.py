import cv2
import numpy as np
import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('my_model.h5')

mapping = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

# Initialize video capture object (webcam)
cap = cv2.VideoCapture(0)  # Change the argument to the video file path if needed

# Check if video capture is opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Define a function to preprocess the frames
def preprocess_frame(frame):
    # Resize the frame to the model's expected input size
    frame_resized = cv2.resize(frame, (224, 224))  # Adjust size as per model's input shape
    # Normalize the pixel values (assuming RGB input, normalize to range [0,1])
    frame_normalized = frame_resized / 255.0
    # Add a batch dimension
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

# Process the video feed
while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video stream or error in reading frame
    
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Predict the class using the model
    predictions = model.predict(preprocessed_frame)
    
    # Get the predicted class index and corresponding probability
    predicted_class_index = np.argmax(predictions)
    predicted_probability = np.max(predictions)
    
    
    # Display the predicted class and probability on the frame
    text = f"Class: {mapping[predicted_class_index]}, Prob: {predicted_probability:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow('Live Video Feed', frame)
    
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
