import cv2
import numpy as np

# ------------------------------------------------------------------------------
# Object Tracking Class
# ------------------------------------------------------------------------------
class TrackedObject:
    """Simple object for tracking centroids across frames."""
    def __init__(self, obj_id, centroid, line_y):
        self.id = obj_id
        self.centroid = centroid          # (cx, cy)
        self.prev_centroid = centroid
        # Determine which side of the line the object is on (0 = above, 1 = below)
        self.side = 0 if centroid[1] < line_y else 1
        self.lost_frames = 0              # count of consecutive frames without detection

    def update(self, centroid, line_y):
        """Update centroid and check for line crossing."""
        self.prev_centroid = self.centroid
        self.centroid = centroid
        new_side = 0 if centroid[1] < line_y else 1
        crossing = False
        # If side changed, a crossing occurred
        if self.side != new_side:
            crossing = True
            self.side = new_side
        self.lost_frames = 0
        return crossing

    def mark_lost(self):
        """Increment lost counter."""
        self.lost_frames += 1


class CentroidTracker:
    """Tracks objects by matching centroids between frames."""
    def __init__(self, max_lost=5, max_distance=50):
        self.next_id = 0
        self.objects = {}                 # id -> TrackedObject
        self.max_lost = max_lost
        self.max_distance = max_distance

    def update(self, centroids, line_y):
        """
        Update tracker with current frame's centroids.
        Returns: set of object IDs that crossed the line in this frame.
        """
        # Mark all existing objects as potentially lost
        for obj in self.objects.values():
            obj.mark_lost()

        matched_ids = set()
        crossed_ids = set()

        # Match centroids to existing objects
        for centroid in centroids:
            best_id = None
            best_dist = self.max_distance
            # Find closest object
            for obj_id, obj in self.objects.items():
                if obj_id in matched_ids:
                    continue
                dist = np.linalg.norm(np.array(obj.centroid) - np.array(centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_id = obj_id
            if best_id is not None:
                obj = self.objects[best_id]
                # Determine side before update
                old_side = 0 if obj.centroid[1] < line_y else 1
                # Update centroid
                obj.prev_centroid = obj.centroid
                obj.centroid = centroid
                new_side = 0 if centroid[1] < line_y else 1
                obj.side = new_side
                obj.lost_frames = 0
                # Check crossing
                if old_side != new_side:
                    crossed_ids.add(best_id)
                matched_ids.add(best_id)
            else:
                # Create new object
                new_obj = TrackedObject(self.next_id, centroid, line_y)
                self.objects[self.next_id] = new_obj
                self.next_id += 1

        # Mark unmatched objects as lost
        for obj_id, obj in self.objects.items():
            if obj_id not in matched_ids:
                obj.lost_frames += 1

        # Remove objects lost for too long
        to_remove = [obj_id for obj_id, obj in self.objects.items()
                     if obj.lost_frames > self.max_lost]
        for obj_id in to_remove:
            del self.objects[obj_id]

        return crossed_ids


# ------------------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------------------
def main():
    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return
    height, width = frame.shape[:2]
    line_y = height // 2   # Horizontal line at the middle

    # Background subtractor (optional, but good for motion detection)
    prev_gray = None

    # Centroid tracker
    tracker = CentroidTracker(max_lost=5, max_distance=50)

    # Counter for line crossings
    count = 0

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Motion detection via frame differencing
        if prev_gray is None:
            prev_gray = gray
            continue

        # Compute absolute difference and threshold
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of moving regions
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Compute centroids of significant contours
        centroids = []
        bounding_rects = []  # for drawing boxes
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Ignore small noise
                continue
            # Bounding box for display
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rects.append((x, y, w, h))
            # Centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        # Update tracker and get IDs that crossed the line in this frame
        crossed_ids = tracker.update(centroids, line_y)

        # Increment counter for each crossing
        for obj_id in crossed_ids:
            count += 1

        # --- Draw visual indicators ---
        # Draw the horizontal line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)

        # Draw bounding boxes around detected motion regions
        for (x, y, w, h) in bounding_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Optionally draw centroids and IDs (for debugging)
        for obj_id, obj in tracker.objects.items():
            cx, cy = obj.centroid
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(obj_id), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display the counter
        cv2.putText(frame, f"Objects Counted: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Motion Crossing Counter", frame)

        # Update previous frame for next iteration
        prev_gray = gray

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()