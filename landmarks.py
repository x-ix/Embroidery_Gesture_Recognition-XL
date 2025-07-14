import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import os
import sys
import pandas as pd

#paths
model_path = 'hand_landmarker.task'

pos_parent_folder = "collected_clips_(positive)"
pos_results_path = "results_(positive)"
pos_output_name = "landmarks_(positive)"

neg_parent_folder = "collected_clips_(negative)"
neg_results_path = "results_(negative)"
neg_output_name = "landmarks_(negative)"

#output file_names
First_Video_name = 'Top-View'
Second_Video_name = 'Bottom-View'

#functionality switches
VISUAL_OUTPUT = True # Whether to save videos with landmarks drawn
MATRIX_OUTPUT = True # Whether to save landmarks as a parquet

#mp config
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#Constants
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
MS_IN_SECOND = 1000
HANDS = 1


# --- Draws hand landmarks and handedness label on the input image ---
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Convert landmarks to protobuf format for drawing
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])

        # Draw landmarks and connections
        solutions.drawing_utils.draw_landmarks(
            annotated_image, proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Calculate text position for handedness label
        h, w, _ = annotated_image.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * w)
        text_y = int(min(y_coords) * h) - MARGIN

        # Draw handedness label on image
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR,
                    FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image



# --- Validates that the dataset path exists and contains subfolders ---
def validate_dataset_structure(dataset_path):
    if not os.path.exists(dataset_path):
        sys.exit("Dataset folder does not exist. Exiting runtime.")

    # List subdirectories in the dataset path
    subfolders = [f for f in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, f))]
    if not subfolders:
        sys.exit("Dataset folder contains no subfolders. Exiting runtime.")
    return subfolders



# --- Creates a DataFrame to store landmark data for each subfolder and video ---
def create_landmark_dataframe(subfolders, video_names):
    df = pd.DataFrame(columns=["name", video_names[0], video_names[1]])
    df["name"] = subfolders
    return df



# --- verify if output folder exists or create if necessary --- 
def ensure_output_folder(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(output_path) and os.listdir(output_path):
        sys.exit("output folder is currently populated which will lead to conflicts, please empty and try again")



# Opens a video file for reading and retrieves its frame rate
def initialise_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps



# Sets up a video writer using capture dimensions and specified codec/fps
def initialise_video_writer(output_path, fourcc, fps, cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer for file: {output_path}")
    return writer



# --- pipeline for generating landmarks and storing in desired formats ---
def pipeline(model, dataset, results_folder_path, dataset_output_name, Input_Video_Names):

    annotated_videos = VISUAL_OUTPUT
    record_landmarks = MATRIX_OUTPUT

        # Exit if no output mode selected
    if not (annotated_videos or record_landmarks):
        sys.exit("No output mode selected. Exiting runtime.")

    # Validate dataset structure and prepare output folder
    subfolders = validate_dataset_structure(dataset)
    subfolders.sort(key=lambda x: int(x))
    ensure_output_folder(results_folder_path)

    # Initialize dataframe for landmarks if recording
    if record_landmarks:
        landmark_df = create_landmark_dataframe(subfolders, Input_Video_Names)
        landmark_df.set_index("name", inplace=True)

    # Configure hand landmarker options
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=HANDS,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Video codec for output videos

    # Process videos in each subfolder and for each video name
    for subfolder in subfolders:
        for video_name in Input_Video_Names:
            position_matrix = [] # Store landmark positions for this video
            video_path = os.path.join(dataset, subfolder, f"{video_name}.mp4")
            cap, fps = initialise_video_capture(video_path)

            # Prepare video writer if saving annotated video
            if annotated_videos:
                output_filename = f"{subfolder}_{video_name}.mp4"
                output_path = os.path.join(results_folder_path, output_filename)
                writer = initialise_video_writer(output_path, fourcc, fps, cap)

            with HandLandmarker.create_from_options(options) as landmarker:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert frame and get timestamp
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    timestamp_ms = ((frame_number / fps) * MS_IN_SECOND) - (MS_IN_SECOND / fps)
                    
                    # Detect hand landmarks
                    detection_result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

                    # Print detected landmarks (for debugging or info)
                    for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        print(f"Hand {hand_index}:")
                        for i, landmark in enumerate(hand_landmarks):
                            print(f"  Landmark {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

                    # Write annotated frame if enabled
                    if annotated_videos:
                        annotated_frame = draw_landmarks_on_image(frame, detection_result)
                        writer.write(annotated_frame)

                    # Record landmarks into position matrix if enabled
                    if record_landmarks and detection_result.hand_landmarks:
                        hand = detection_result.hand_landmarks[0]  # one hand per camera angle
                        matrix_entry = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                        matrix_entry = matrix_entry.flatten()
                        position_matrix.append(matrix_entry)
            
            # Save recorded landmarks to dataframe per video
            if record_landmarks:
                landmark_df.at[subfolder, video_name] = position_matrix
                print(landmark_df)

            # Release resources
            cap.release()
            if annotated_videos:
                writer.release()

    # Save landmarks dataframe to disk if matrix_output is True
    if record_landmarks:
        json_path = os.path.join(results_folder_path, f"{dataset_output_name}.json")
        parquet_path = os.path.join(results_folder_path, f"{dataset_output_name}.parquet")
        landmark_df.to_json(json_path)
        landmark_df.to_parquet(parquet_path)


def main():
    #initialises names for existing video pairs
    input_Video_Names = [First_Video_name, Second_Video_name]

    #pipeline run on positive instances
    pipeline(model_path, pos_parent_folder, pos_results_path, pos_output_name, input_Video_Names)

    #pipeline run on negative instances
    pipeline(model_path, neg_parent_folder, neg_results_path, neg_output_name, input_Video_Names)



if __name__ == "__main__":
    main()