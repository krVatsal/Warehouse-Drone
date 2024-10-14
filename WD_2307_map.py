import cv2
import numpy as np
import argparse

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

class Arena:
    def __init__(self, image_path, aruco_type="DICT_4X4_250"):
        self.image_path = image_path
        self.aruco_type = aruco_type
        self.detected_markers = []
        self.obstacles = 0
        self.total_area = 0
        self.aruco_ids = []
        self.arena_area = 0

    def aruco_display(self, corners, ids, rejected, image):
        if len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                
                cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print(f"[Inference] ArUco marker ID: {markerID}")
        
        return image

    def four_point_transform(self, image, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def resize_for_display(self, image, max_width=1200, max_height=800):
        h, w = image.shape[:2]
        if h > max_height or w > max_width:
            scale = min(max_height/h, max_width/w)
            new_size = (int(w*scale), int(h*scale))
            return cv2.resize(image, new_size)
        return image

    def identification(self):
        # Step 1: Read the image
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"Error: Unable to load image {self.image_path}")
            return

        # Step 2: Detect ArUco markers
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[self.aruco_type])
        arucoParams = cv2.aruco.DetectorParameters_create()

        corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
        
        print("Detected IDs:", ids)
        print("Number of corners detected:", len(corners))
        
        if ids is None or len(ids) < 4:
            print("Error: Could not detect 4 ArUco markers.")
            return

        self.aruco_ids = ids.flatten().tolist()

        # Display detected markers
        detected_markers = self.aruco_display(corners, ids, rejected, image.copy())
        # cv2.imshow("Detected ArUco Markers", self.resize_for_display(detected_markers))

        # Step 3: Extract corner points for perspective transform
        try:
            if len(corners) != 4:
                raise ValueError(f"Expected 4 markers, but found {len(corners)}")
            
            corner_points = np.array([corner.reshape(4, 2)[0] for corner in corners])
            
            center = np.mean(corner_points, axis=0)
            corner_points = sorted(corner_points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
            corner_points = np.array(corner_points, dtype=np.float32)

            print("Corner points for transform:", corner_points)
        except Exception as e:
            print(f"Error processing corners: {e}")
            print("Corners:", corners)
            print("IDs:", ids)
            return

        # Step 4: Apply perspective transform
        try:
            warped = self.four_point_transform(image, corner_points)
            self.arena_area = warped.shape[0] * warped.shape[1]
            print(f"Arena dimensions: {warped.shape[1]}x{warped.shape[0]}")
            print(f"Total arena area: {self.arena_area} pixels")

            # Resize warped image to match original dimensions
            original_height, original_width = image.shape[:2]
            warped = cv2.resize(warped, (original_width, original_height))
        except Exception as e:
            print(f"Error applying perspective transform: {e}")
            return
        
        # Step 5: Preprocess the warped image
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_warped, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Step 6: Detect obstacles
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 12000  # Adjust this value based on your image scale
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        self.obstacles = len(filtered_contours)
        self.total_area = sum(cv2.contourArea(c) for c in filtered_contours)

        print(f"Number of obstacles: {self.obstacles}")
        print(f"Total area of obstacles: {self.total_area} pixels")
        print(f"Percentage of arena covered by obstacles: {(self.total_area / self.arena_area) * 100:.2f}%")

        # Visualize the results
        result_image = warped.copy()
        cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)
        
        for i, contour in enumerate(filtered_contours):
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result_image, f"{i+1}: {area:.0f}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # cv2.imshow("Warped Arena", self.resize_for_display(warped))
        # cv2.imshow("Thresholded Image", self.resize_for_display(thresh))
        # cv2.imshow("Detected Obstacles", self.resize_for_display(result_image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def text_file(self):
        with open("obstacles.txt", "w") as file:
            file.write(f"ArUco IDs: [{', '.join(map(str, self.aruco_ids))}]\n")
            file.write(f"Obstacles: {self.obstacles}\n")
            file.write(f"Area: {self.total_area}\n")
            # file.write(f"Arena Area: {self.arena_area} pixels\n")
            # file.write(f"Percentage of Arena Covered: {(self.total_area / self.arena_area) * 100:.2f}%\n")
            

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image to be processed")
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_250", help="Type of ArUco tag to detect")
    args = vars(ap.parse_args())

    image_path = args['image']
    aruco_type = args['type']

    arena = Arena(image_path, aruco_type)
    arena.identification()
    arena.text_file()