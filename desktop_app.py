import cv2
import numpy as np
import time
from ascii_view.image import to_grayscale
from ascii_view.linalg import svd_compress, project_matrix, sobel_edges, combine_matrices
from ascii_view.cv_render import render_cv_fast

# Default Configuration Defaults
DEFAULT_WIDTH = 150
DEFAULT_RANK = 30
DEFAULT_EDGE_WT = 40  # 0-100 representing 0.0 to 1.0

def main():
    print("Initializing Webcam...")
    cap = cv2.VideoCapture(0)
    
    # Try to set high resolution for sharp source before we downscale
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Warm up camera
    for _ in range(5):
        cap.read()
        
    print("Webcam initialized. Press 'q' or 'ESC' to quit.")
    
    # ── Windows & Controls ──────────────────────────────────────────────────
    cv2.namedWindow("ASCIIfy | Native OpenCV", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 500, 200)

    def noop(x): 
        pass

    cv2.createTrackbar("Width", "Controls", DEFAULT_WIDTH, 400, noop)
    cv2.createTrackbar("SVD Rank", "Controls", DEFAULT_RANK, 200, noop)
    cv2.createTrackbar("Edge Weight", "Controls", DEFAULT_EDGE_WT, 100, noop)

    char_ratio = 10.0 / 6.0 
    
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Read controls
        current_width = max(40, cv2.getTrackbarPos("Width", "Controls"))
        current_rank  = max(1, cv2.getTrackbarPos("SVD Rank", "Controls"))
        current_edge  = cv2.getTrackbarPos("Edge Weight", "Controls") / 100.0
            
        # 1. Resize for ASCII grid
        orig_h, orig_w, _ = frame.shape
        
        # Maintain aspect ratio natively
        new_w = current_width
        new_h = max(1, int(current_width * (orig_h / orig_w) / char_ratio))
        
        small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Math Pipeline
        img_f = small_frame.astype(np.float64)
        grayscale = to_grayscale(img_f)
        compressed, U, S, Vt = svd_compress(grayscale, current_rank)
        projected, _error = project_matrix(grayscale, U, S, Vt)
        edges, edge_dirs = sobel_edges(projected, return_dirs=True)
        combined = combine_matrices(projected, edges, current_edge)

        # 3. Super Fast Rendering (Numpy Vectorized)
        ascii_img = render_cv_fast(small_frame, combined, edges=edges, edge_dirs=edge_dirs)
        
        # Show FPS and stats
        fps = 1.0 / (time.time() - t0)
        status_text = f"FPS: {fps:.1f} | Width: {current_width} | Rank: {current_rank} | Edge: {current_edge:.2f}"
        cv2.putText(ascii_img, status_text, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("ASCIIfy | Native OpenCV", ascii_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
