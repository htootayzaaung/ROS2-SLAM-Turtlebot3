import os
import gdown
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class_labels = {
    0: "Background",
    1: "Earth",
    2: "Jupiter",
    3: "MakeMake",
    4: "Mars",
    5: "Mercury",
    6: "Moon",
    7: "Neptune",
    8: "Pluto",
    9: "Saturn",
    10: "Uranus",
    11: "Venus"
}

# Define the model directory and file path
model_directory = os.path.join(os.path.expanduser('~/ros2_ws/src/group_project'), 'models')
model_file_path = os.path.join(model_directory, 'fasterrcnn_resnet50_fpn_epoch14.pth')

# Define the model URL from Google Drive
model_url = 'https://drive.google.com/uc?id=1Xt-O39jteVWv2CW9IRphZ62mt4cykkBp'

def setup_model_directory():
    # Create the model directory if it doesn't exist
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        print(f"Created model directory at {model_directory}")
    else:
        print(f"Model directory already exists at {model_directory}")

def download_model():
    # Check if the model file already exists
    if not os.path.isfile(model_file_path):
        print(f"Model file not found, downloading from {model_url}")
        gdown.download(model_url, model_file_path, quiet=False)
    else:
        print(f"Model file already exists at {model_file_path}")

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.model = self.load_model()
        self.model.eval()  # Set the model to evaluation mode

    def load_model(self):
        # Load model with weights specification
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        num_classes = 12  # Example, adjust accordingly
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Load the state dict
        state_dict = torch.load(model_file_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Convert the OpenCV image from BGR to RGB format
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert the OpenCV RGB image to a tensor
        input_tensor = F.to_tensor(cv_image_rgb)
        # Ensure the tensor is on the appropriate device
        input_tensor = input_tensor.to('cpu')

        # Perform object detection - ensure input_tensor is within a list
        with torch.no_grad():
            prediction = self.model([input_tensor])[0]

        # Draw bounding boxes and labels on the original cv_image (BGR format)
        self.display_image(cv_image, prediction)

    def display_image(self, cv_image, prediction):
        # Define a confidence threshold
        confidence_threshold = 0.8

        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            # Convert to numpy array and get the confidence score
            box = box.cpu().numpy().astype(int)
            score = score.cpu().numpy()

            # Check if the detection confidence is above the threshold
            if score > confidence_threshold:
                # Combine class label and confidence score in the label text
                label_text = f"{class_labels[label.item()]}: {score:.2f}"
                # Draw the bounding box
                cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Calculate the position for the label text (inside the bounding box)
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_position = (max(box[0], label_size[0]), max(box[1] + label_size[1], label_size[1] + 5))
                
                # Draw the label background
                cv2.rectangle(cv_image, (box[0], box[1]), (box[0] + label_size[0], box[1] + label_size[1] + 5), (0, 255, 0), cv2.FILLED)
                # Put the label text
                cv2.putText(cv_image, label_text, (box[0], box[1] + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Display the annotated image
        cv2.imshow('Object Detection', cv_image)
        cv2.waitKey(1)
def main():
    setup_model_directory()
    download_model()

    # Initialize ROS
    rclpy.init(args=None)
    object_detector = ObjectDetector()
    rclpy.spin(object_detector)

    object_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
