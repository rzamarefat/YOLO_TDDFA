from TDDFA import TDDFA
import yaml
from ultralytics import YOLO
from align_trans import warp_and_crop_face, get_reference_facial_points
from config import YOLO_TDDFA_CONFIG

class YOLO_TDDFA:
    def __init__(self):
        cfg = yaml.load(open(YOLO_TDDFA_CONFIG["path_to_yaml_config"]), Loader=yaml.SafeLoader)
        self._tddfa = TDDFA(gpu_mode=False, **cfg)
        self._yolo_model = YOLO(YOLO_TDDFA_CONFIG["path_to_yolo_ckpt"])
        self.yolo_confidence_threshold = YOLO_TDDFA_CONFIG["yolo_confidence_threshold"] 


    def _find_yaw(self, points):
        """
        Parameters
        ----------
        points : TYPE - Array of float32, Size = (10,)
            coordinates of landmarks for the selected faces.
        Returns
        -------
        TYPE
            yaw of face.
        """
        le2n = points[2] - points[0]
        re2n = points[1] - points[2]
        return le2n - re2n

    def detect(self, image):
        faces = []
        yaws = []

        results = self._yolo_model(image)
        boxes = results[0].boxes

        converted_boxes = []
        for box in boxes:
            if box.conf.item() > self.yolo_confidence_threshold:
                top_left_x = int(box.xyxy.tolist()[0][0])
                top_left_y = int(box.xyxy.tolist()[0][1])
                bottom_right_x = int(box.xyxy.tolist()[0][2])
                bottom_right_y = int(box.xyxy.tolist()[0][3])

                converted_boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
            
        param_lst, roi_box_lst = self._tddfa(image, converted_boxes)
        dense_flag = False
        ver_lst = self._tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

        landmarks = []
        for i in range(len(ver_lst)):
            landmarks.append([[ver_lst[i][0][41], ver_lst[i][1][41]],  
                                [ver_lst[i][0][47], ver_lst[i][1][47]],
                                [ver_lst[i][0][30], ver_lst[i][1][30]],
                                [ver_lst[i][0][59], ver_lst[i][1][59]],
                                [ver_lst[i][0][64], ver_lst[i][1][64]]
                            ])
    
        reference_pts = get_reference_facial_points(default_square=True)
        for landmark in landmarks:

            face = warp_and_crop_face(image,
                        landmark,
                        reference_pts=reference_pts,
                        align_type='smilarity')


            landmark_for_yaw = [landmark[0][0], landmark[1][0], landmark[2][0], landmark[3][0], landmark[4][0],
                                landmark[0][1], landmark[1][1], landmark[2][1], landmark[3][1], landmark[4][1],
                                ]
            yaw = self._find_yaw(landmark_for_yaw)

            faces.append(face)
            yaws.append(yaw)
        
        print(yaws)
        return faces, converted_boxes, yaws


if __name__ == "__main__":
    import cv2

    yolo_tddfa = YOLO_TDDFA()
    
    img = cv2.imread("/home/mehran/rezamarefat/ATTENDANCE_DB_2/2023-07-22-11-03-28-892502.jpg")

    faces, boxes, yaws = yolo_tddfa.detect(img)

    for index, (face, box, yaw) in enumerate(zip(faces, boxes, yaws)):
        cv2.imwrite(f"face__{index}.jpg", face)
        print(index, "yaw", yaw)

    
