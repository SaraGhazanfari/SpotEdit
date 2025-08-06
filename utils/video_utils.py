import cv2


def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def save_frames_as_video(frames, original_video_path, output_path):
    cap = cv2.VideoCapture(original_video_path)

    # Get original video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap.release()

    # Define the codec and create VideoWriter object
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Grayscale-friendly
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    for frame in frames:
        # Resize if needed (to avoid size mismatch)
        frame_resized = cv2.resize(frame, (width, height))
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)
        out.write(frame_resized)

    out.release()