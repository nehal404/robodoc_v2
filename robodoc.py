import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QGroupBox, QGridLayout, QMessageBox, QDialog, QRubberBand,
    QScrollArea, QFrame, QTextEdit, QSplitter
)
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QPalette
from PIL import Image
import traceback
import os

# Import AI prediction function
try:
    from tfmodel import predict_image
except ImportError:
    def predict_image(image_path):
        # Fallback function for demo purposes
        return "Demo Mode", 0.95

DISPLAY_W, DISPLAY_H = 500, 250  # Larger display area for professional appearance

class_names = {
    0: "First Degree Burn",
    1: "Second Degree Burn", 
    2: "Third Degree Burn",
    3: "Lacerations",
    4: "Skin Ulcer Wound",
    5: "No Wound/Healthy Skin"
}

def cvimg_to_qpixmap(img, w=DISPLAY_W, h=DISPLAY_H):
    """Safely convert CV2 image to QPixmap with proper error handling"""
    try:
        if img is None:
            return QPixmap()
        
        # Ensure image is in correct format
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        
        # Convert to PIL for reliable resizing
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
        
        # Convert to QImage
        data = pil_img.tobytes()
        qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGB888)
        
        return QPixmap.fromImage(qimg)
    except Exception as e:
        print(f"Error converting image to pixmap: {e}")
        return QPixmap()

def safe_image_to_qpixmap(cv_img, max_width=800, max_height=600):
    """Safely convert CV2 image to QPixmap for display"""
    try:
        if cv_img is None:
            return QPixmap(), (0, 0)
        
        h, w = cv_img.shape[:2]
        original_size = (h, w)
        
        # Calculate scaling to fit within max dimensions
        scale_w = max_width / w if w > max_width else 1.0
        scale_h = max_height / h if h > max_height else 1.0
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert color space safely
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif len(cv_img.shape) == 2:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = cv_img
        
        # Create QImage
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        return QPixmap.fromImage(qimg), original_size
    except Exception as e:
        print(f"Error in safe_image_to_qpixmap: {e}")
        traceback.print_exc()
        return QPixmap(), (0, 0)

def findcontours(image1, image2, threshvalue, linestep):
    """Enhanced contour finding with improved processing"""
    try:
        if image1 is None or image2 is None:
            raise ValueError("One or both images are None")
        
        # Resize image2 to match image1 dimensions
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        original = image1.copy()
        
        # Convert to grayscale safely
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2
        
        # Enhanced difference processing
        difference = cv2.absdiff(gray_image2, gray_image1)
        blur = cv2.GaussianBlur(difference, (11, 11), 0)
        
        # Automatic threshold (Otsu method)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations for noise removal
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours with area filtering
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 500]  # Filter small contours
        
        # Draw contours on original image
        contoured = original.copy()
        cv2.drawContours(contoured, contours, -1, (0, 255, 0), 3)
        
        # Create mask and apply it
        mask = np.zeros_like(original)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(original, mask)
        
        # Generate adaptive spray pattern
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        line_coordinates = []
        for x in range(result_gray.shape[1]):
            col_pixels = result_gray[:, x]
            if cv2.countNonZero(col_pixels) != 0:
                y_coordinates = np.where(col_pixels > 0)[0]
                line_coordinates.append([(x, y) for y in y_coordinates])
        
        # Create spray lines with adaptive density
        step = max(5, min(30, int(result_gray.shape[1] / 50)))
        line_coordinates_np = [np.array(line, dtype=np.int32) for line in line_coordinates]
        lines = [line_coordinates_np[i] for i in range(0, len(line_coordinates_np), step)]
        
        # Draw spray pattern
        lined = contoured.copy()
        if lines:
            cv2.polylines(lined, lines, isClosed=False, color=(0, 255, 0), thickness=2)
        
        # Combine results: Original | Contoured | Processed
        combined = np.hstack([original, contoured, lined])
        return combined, contours
    except Exception as e:
        print(f"Error in findcontours: {e}")
        traceback.print_exc()
        raise

def save_selected_contours(image, contours, filepath):
    """Save isolated contours as transparent PNG"""
    try:
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255,255,255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(image, mask)
        
        # Create transparent PNG
        b, g, r = cv2.split(result)
        alpha = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        rgba = cv2.merge([b, g, r, alpha])
        cv2.imwrite(filepath, rgba)
    except Exception as e:
        print(f"Error saving contours: {e}")
        raise

class CropDialog(QDialog):
    """Professional crop dialog for control region selection"""
    def __init__(self, cv_image):
        super().__init__()
        self.setWindowTitle("Select Control Region - RoboDoc v2")
        self.cv_image = cv_image.copy()
        self.crop_rect = None
        self.is_cropping = False
        
        try:
            self.original_h, self.original_w = cv_image.shape[:2]
            self.display_pixmap, _ = safe_image_to_qpixmap(cv_image, 800, 600)
            
            if self.display_pixmap.isNull():
                raise ValueError("Failed to create display pixmap")
            
            self.scale_x = self.original_w / self.display_pixmap.width()
            self.scale_y = self.original_h / self.display_pixmap.height()
            
            self.setup_ui()
            
        except Exception as e:
            print(f"Error initializing CropDialog: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to initialize crop dialog: {str(e)}")
            self.reject()

    def setup_ui(self):
        try:
            self.scroll_area = QScrollArea()
            self.image_label = QLabel()
            self.image_label.setPixmap(self.display_pixmap)
            self.image_label.setScaledContents(False)
            
            self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)
            self.origin = QPoint()
            
            self.image_label.mousePressEvent = self.mouse_press_event
            self.image_label.mouseMoveEvent = self.mouse_move_event
            self.image_label.mouseReleaseEvent = self.mouse_release_event
            
            self.scroll_area.setWidget(self.image_label)
            self.scroll_area.setWidgetResizable(True)
            
            layout = QVBoxLayout()
            
            # Professional header
            header_label = QLabel("Select Healthy Control Region")
            header_font = QFont()
            header_font.setPointSize(16)
            header_font.setBold(True)
            header_label.setFont(header_font)
            header_label.setStyleSheet("color: #2c3e50; padding: 10px;")
            layout.addWidget(header_label)
            
            instruction_label = QLabel("Drag to select a healthy region from the injury image for comparison baseline.")
            instruction_label.setWordWrap(True)
            instruction_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-style: italic;")
            layout.addWidget(instruction_label)
            
            info_label = QLabel(f"Original: {self.original_w}×{self.original_h} | Display: {self.display_pixmap.width()}×{self.display_pixmap.height()}")
            info_label.setStyleSheet("color: #34495e; font-size: 10px; padding: 2px;")
            layout.addWidget(info_label)
            
            layout.addWidget(self.scroll_area)
            
            # Professional buttons
            button_layout = QHBoxLayout()
            ok_btn = QPushButton("Accept Selection")
            cancel_btn = QPushButton("Cancel")
            
            ok_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #3498db, stop: 1 #2980b9);
                    border: 1px solid #2980b9;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: white;
                    font-weight: bold;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #5dade2, stop: 1 #3498db);
                }
            """)
            
            cancel_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #e74c3c, stop: 1 #c0392b);
                    border: 1px solid #c0392b;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: white;
                    font-weight: bold;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #ec7063, stop: 1 #e74c3c);
                }
            """)
            
            ok_btn.clicked.connect(self.accept)
            cancel_btn.clicked.connect(self.reject)
            button_layout.addStretch()
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            self.setLayout(layout)
            
            dialog_width = min(900, self.display_pixmap.width() + 50)
            dialog_height = min(700, self.display_pixmap.height() + 150)
            self.resize(dialog_width, dialog_height)
            
        except Exception as e:
            print(f"Error setting up UI: {e}")
            traceback.print_exc()
            raise

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            self.is_cropping = True

    def mouse_move_event(self, event):
        if self.is_cropping:
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton and self.is_cropping:
            self.crop_rect = self.rubber_band.geometry()
            self.is_cropping = False

    def get_crop_coordinates(self):
        """Return crop coordinates in original image space"""
        if self.crop_rect is None or self.crop_rect.isEmpty():
            return None
            
        x = max(0, int(self.crop_rect.x() * self.scale_x))
        y = max(0, int(self.crop_rect.y() * self.scale_y))
        w = int(self.crop_rect.width() * self.scale_x)
        h = int(self.crop_rect.height() * self.scale_y)
        
        x = min(x, self.original_w - 1)
        y = min(y, self.original_h - 1)
        w = min(w, self.original_w - x)
        h = min(h, self.original_h - y)
        
        return (x, y, w, h)

class RoboDocV2App(QWidget):
    """Enhanced RoboDoc v2 Application with Professional UI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboDoc v2 - Real-Time Injury Classification & Analysis")
        self.injury_img = None
        self.cropped_control = None
        self.result_img = None
        self.contours = None
        self.threshvalue = 50
        self.linestep = 20
        self.ai_prediction = None
        self.ai_confidence = 0.0
        
        self.setup_ui()
        self.apply_professional_styling()

    def create_header(self):
        """Create professional header with logo and title"""
        header_frame = QFrame()        
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)
        
        # Logo section
        logo_label = QLabel()
        try:
            pixmap = QPixmap("robodoc_v2/logos/robodoc_logo.png")  # Adjust path as needed
            if not pixmap.isNull():
                logo_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(logo_pixmap)
            else:
                logo_label.setText("LOGO")
                logo_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold; border: 2px solid white; padding: 20px;")
        except:
            logo_label.setText("RoboDoc")
            logo_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; border: 2px solid white; padding: 15px;")
        
        logo_label.setAlignment(Qt.AlignCenter)
        
        # Title section
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        title_label = QLabel("RoboDoc v2")
        title_font = QFont()
        title_font.setPointSize(40)
        title_font.setBold(True)
        title_font.setFamily("Arial")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the title

        subtitle_label = QLabel("AI-Powered Medical Imaging • Version 2/7")
        subtitle_font = QFont()
        subtitle_font.setPointSize(24)
        subtitle_font.setFamily("Arial")
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the subtitle
        
        title_layout.addWidget(logo_label)
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.addStretch()
        
        header_layout.addLayout(title_layout)
        
        header_frame.setLayout(header_layout)
        return header_frame
        
    

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)
        
        # Professional header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results and AI Analysis
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 2)  # Controls take 2/3
        splitter.setStretchFactor(1, 1)  # Results take 1/3
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_control_panel(self):
        """Create the control panel with all input controls"""
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_layout.setSpacing(15)
        
        # Step 1: Upload injury image
        step1_group = QGroupBox("Step 1: Upload Injury Image")
        step1_group.setStyleSheet(self.get_groupbox_style())
        step1_layout = QHBoxLayout()
        
        self.injury_label = QLabel("No injury image selected")
        self.injury_label.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 5px;")
        self.upload_injury_btn = QPushButton("Upload Injury Image")
        self.upload_injury_btn.setStyleSheet(self.get_button_style())
        
        step1_layout.addWidget(self.injury_label, 1)
        step1_layout.addWidget(self.upload_injury_btn)
        step1_group.setLayout(step1_layout)
        control_layout.addWidget(step1_group)

        # Step 2: Select control region
        step2_group = QGroupBox("Step 2: Select Control Region")
        step2_group.setStyleSheet(self.get_groupbox_style())
        step2_layout = QHBoxLayout()
        
        step2_info = QLabel("Select a healthy region from the injury image for comparison baseline.")
        step2_info.setWordWrap(True)
        step2_info.setStyleSheet("color: #34495e; padding: 5px;")
        
        self.crop_btn = QPushButton("Select Control Region")
        self.crop_btn.setEnabled(False)
        self.crop_btn.setStyleSheet(self.get_button_style(disabled=True))
        
        step2_layout.addWidget(step2_info, 1)
        step2_layout.addWidget(self.crop_btn)
        step2_group.setLayout(step2_layout)
        control_layout.addWidget(step2_group)

        # Step 3: Parameters
        step3_group = QGroupBox("Step 3: Adjust Detection Parameters")
        step3_group.setStyleSheet(self.get_groupbox_style())
        param_layout = QGridLayout()
        param_layout.setSpacing(15)
        
        # Threshold parameter
        thresh_label = QLabel("Injury Detection Sensitivity:")
        thresh_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setMinimum(3)
        self.thresh_slider.setMaximum(190)
        self.thresh_slider.setValue(self.threshvalue)
        self.thresh_slider.valueChanged.connect(self.update_thresh)
        self.thresh_slider.setStyleSheet(self.get_slider_style())
        
        self.thresh_value_label = QLabel(str(self.threshvalue))
        self.thresh_value_label.setStyleSheet("font-weight: bold; color: #e74c3c; min-width: 30px; padding: 2px;")
        
        # Line density parameter
        line_label = QLabel("Spray Pattern Tightness:")
        line_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        
        self.line_slider = QSlider(Qt.Horizontal)
        self.line_slider.setMinimum(1)
        self.line_slider.setMaximum(50)
        self.line_slider.setValue(self.linestep)
        self.line_slider.valueChanged.connect(self.update_line)
        self.line_slider.setStyleSheet(self.get_slider_style())
        
        self.line_value_label = QLabel(str(self.linestep))
        self.line_value_label.setStyleSheet("font-weight: bold; color: #e74c3c; min-width: 30px; padding: 2px;")
        
        param_layout.addWidget(thresh_label, 0, 0)
        param_layout.addWidget(self.thresh_slider, 0, 1)
        param_layout.addWidget(self.thresh_value_label, 0, 2)
        param_layout.addWidget(line_label, 1, 0)
        param_layout.addWidget(self.line_slider, 1, 1)
        param_layout.addWidget(self.line_value_label, 1, 2)
        
        step3_group.setLayout(param_layout)
        control_layout.addWidget(step3_group)

        # Step 4: Analysis
        step4_group = QGroupBox("Step 4: AI Analysis & Export")
        step4_group.setStyleSheet(self.get_groupbox_style())
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(10)
        
        # Analysis button
        self.analyze_btn = QPushButton("Analyze & Classify Injury")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet(self.get_button_style(disabled=True, primary=True))
        
        # Export buttons layout
        export_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Analysis")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(self.get_button_style(disabled=True))
        
        self.save_contour_btn = QPushButton("Export Contours")
        self.save_contour_btn.setEnabled(False)
        self.save_contour_btn.setStyleSheet(self.get_button_style(disabled=True))
        
        export_layout.addWidget(self.save_btn)
        export_layout.addWidget(self.save_contour_btn)
        
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addLayout(export_layout)
        step4_group.setLayout(btn_layout)
        control_layout.addWidget(step4_group)
        
        control_layout.addStretch()
        control_widget.setLayout(control_layout)
        
        # Connect signals
        self.upload_injury_btn.clicked.connect(self.upload_injury)
        self.crop_btn.clicked.connect(self.crop_control)
        self.analyze_btn.clicked.connect(self.analyze_injury)
        self.save_btn.clicked.connect(self.save_result)
        self.save_contour_btn.clicked.connect(self.save_selected_contours)
        
        return control_widget

    def create_results_panel(self):
        """Create the results panel with image display and AI analysis"""
        results_widget = QWidget()
        results_layout = QVBoxLayout()
        results_layout.setSpacing(15)
        
        # Analysis Results Display
        display_group = QGroupBox("Analysis Results")
        display_group.setStyleSheet(self.get_groupbox_style())
        display_layout = QVBoxLayout()
        
        self.result_label = QLabel()
        self.result_label.setFixedSize(DISPLAY_W, DISPLAY_H)
        self.result_label.setStyleSheet("""
            border: 2px dashed #bdc3c7; 
            background-color: #f8f9fa;
            border-radius: 8px;
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setText("Analysis results will appear here\n\nOriginal | Contoured | Processed")
        
        display_layout.addWidget(self.result_label)
        display_group.setLayout(display_layout)
        results_layout.addWidget(display_group)
        
        # AI Classification Results
        ai_group = QGroupBox("AI Classification Results")
        ai_group.setStyleSheet(self.get_groupbox_style())
        ai_layout = QVBoxLayout()
        ai_layout.setSpacing(10)
        
        # AI Prediction Display
        self.ai_result_label = QLabel("AI Prediction: Awaiting Analysis")
        self.ai_result_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            background-color: #ecf0f1;
        """)
        self.ai_result_label.setAlignment(Qt.AlignCenter)
        
        # Confidence Display
        self.confidence_label = QLabel("Confidence: ---%")
        self.confidence_label.setStyleSheet("""
            font-size: 12px;
            color: #7f8c8d;
            padding: 5px;
            text-align: center;
        """)
        self.confidence_label.setAlignment(Qt.AlignCenter)
        
        # Classification Guide
        guide_label = QLabel("Classification Categories:")
        guide_label.setStyleSheet("font-weight: bold; color: #2c3e50; margin-top: 10px;")
        
        self.classification_guide = QTextEdit()
        self.classification_guide.setMaximumHeight(150)
        self.classification_guide.setStyleSheet("""
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 5px;
                background-color: #f8f9fa;
                font-size: 10px;
            }
        """)
        
        guide_text = """
0 - First Degree Burn: Superficial burns affecting only outer skin layer
1 - Second Degree Burn: Burns affecting outer and underlying skin layers  
2 - Third Degree Burn: Severe burns through all skin layers
3 - Lacerations: Cuts or tears in the skin tissue
4 - Skin Ulcer Wound: Open sores on skin surface
5 - No Wound/Healthy Skin: Normal healthy tissue without injury
        """
        
        self.classification_guide.setPlainText(guide_text)
        self.classification_guide.setReadOnly(True)
        
        ai_layout.addWidget(self.ai_result_label)
        ai_layout.addWidget(self.confidence_label)
        ai_layout.addWidget(guide_label)
        ai_layout.addWidget(self.classification_guide)
        
        ai_group.setLayout(ai_layout)
        results_layout.addWidget(ai_group)
        
        results_layout.addStretch()
        results_widget.setLayout(results_layout)
        return results_widget

    def get_groupbox_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                margin-top: 10px;
                padding: 15px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                background-color: #ffffff;
                color: #2c3e50;
            }
        """

    def get_button_style(self, disabled=False, primary=False):
        if disabled:
            return """
                QPushButton {
                    background-color: #ecf0f1;
                    border: 1px solid #bdc3c7;
                    border-radius: 6px;
                    padding: 10px 16px;
                    font-size: 12px;
                    color: #95a5a6;
                    min-width: 140px;
                }
            """
        elif primary:
            return """
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #3498db, stop: 1 #2980b9);
                    border: 1px solid #2980b9;
                    border-radius: 6px;
                    padding: 10px 16px;
                    font-size: 12px;
                    font-weight: bold;
                    color: white;
                    min-width: 140px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #5dade2, stop: 1 #3498db);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #2980b9, stop: 1 #21618c);
                }
            """
        else:
            return """
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #ffffff, stop: 1 #f8f9fa);
                    border: 1px solid #bdc3c7;
                    border-radius: 6px;
                    padding: 10px 16px;
                    font-size: 12px;
                    color: #2c3e50;
                    min-width: 140px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #f8f9fa, stop: 1 #e9ecef);
                    border: 1px solid #95a5a6;
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #e9ecef, stop: 1 #dee2e6);
                }
            """

    def get_slider_style(self):
        return """
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 8px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #ecf0f1, stop: 1 #d5dbdb);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #3498db, stop: 1 #2980b9);
                border: 1px solid #2980b9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #5dade2, stop: 1 #3498db);
            }
        """

    def apply_professional_styling(self):
        """Apply professional medical interface styling"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                background-color: #f5f6fa;
            }
            QLabel {
                color: #2c3e50;
            }
            QSplitter::handle {
                background-color: #bdc3c7;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #95a5a6;
            }
        """)
        self.setMinimumSize(1200, 900)
        self.resize(1400, 1000)

    def upload_injury(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Injury Image - RoboDoc v2", "", 
                "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            
            if not path:
                return
                
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Error", "Could not load image. Please check the file format.")
                return
            
            if len(img.shape) != 3 or img.shape[2] != 3:
                QMessageBox.warning(self, "Error", "Please select a color image (BGR/RGB format).")
                return
                
            self.injury_img = img
            self.current_image_path = path  # Store for AI analysis
            filename = path.split('/')[-1] if '/' in path else path.split('\\')[-1]
            self.injury_label.setText(f"✓ {filename} ({img.shape[1]}×{img.shape[0]})")
            self.injury_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
            
            # Enable crop button
            self.crop_btn.setEnabled(True)
            self.crop_btn.setStyleSheet(self.get_button_style())
            
            # Reset state
            self.reset_analysis_state()
            self.result_label.setText("Image loaded successfully!\n\nProceed to select control region")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            print(f"Upload error: {e}")
            traceback.print_exc()

    def crop_control(self):
        if self.injury_img is None:
            return
            
        try:
            dialog = CropDialog(self.injury_img)
            
            if dialog.exec_() == QDialog.Accepted:
                coords = dialog.get_crop_coordinates()
                
                if coords is None:
                    QMessageBox.warning(self, "No Selection", "Please select a region to crop.")
                    return
                
                x, y, w, h = coords
                
                if w <= 10 or h <= 10:
                    QMessageBox.warning(self, "Invalid Selection", "Selected region is too small.")
                    return
                
                self.cropped_control = self.injury_img[y:y+h, x:x+w].copy()
                
                if self.cropped_control.size == 0:
                    QMessageBox.warning(self, "Crop Error", "Failed to crop image.")
                    return
                
                # Enable analysis button
                self.analyze_btn.setEnabled(True)
                self.analyze_btn.setStyleSheet(self.get_button_style(primary=True))
                
                self.result_label.setText(f"Control region selected: {w}×{h} pixels\n\nReady for AI analysis")
                QMessageBox.information(self, "Success", 
                                      f"Control region selected: {w}×{h} pixels\nYou can now proceed with AI analysis.")
                
        except Exception as e:
            QMessageBox.critical(self, "Crop Error", f"Failed to crop image: {str(e)}")
            print(f"Crop error: {e}")
            traceback.print_exc()

    def analyze_injury(self):
        """Perform comprehensive injury analysis with AI classification"""
        if self.injury_img is None or self.cropped_control is None:
            return
            
        try:
            # Update UI to show processing
            self.ai_result_label.setText("AI Analysis: Processing...")
            self.ai_result_label.setStyleSheet("""
                font-size: 14px;
                font-weight: bold;
                color: #f39c12;
                padding: 10px;
                border: 1px solid #f39c12;
                border-radius: 5px;
                background-color: #fef9e7;
            """)
            self.confidence_label.setText("Please wait...")
            self.result_label.setText("Processing analysis...\nPlease wait...")
            QApplication.processEvents()
            
            # Step 1: Computer vision analysis
            combined, contours = findcontours(
                self.injury_img, self.cropped_control, 
                self.threshvalue, self.linestep
            )
            
            self.result_img = combined
            self.contours = contours
            
            # Display computer vision results
            pixmap = cvimg_to_qpixmap(combined)
            self.result_label.setPixmap(pixmap)
            
            # Step 2: AI Classification
            try:
                # Save injury image for AI analysis
                save_path = "temp_analysis"
                os.makedirs(save_path, exist_ok=True)
                temp_image_path = os.path.join(save_path, "temp_injury.jpg")
                cv2.imwrite(temp_image_path, self.injury_img)
                
                # Get AI prediction
                prediction_result = predict_image(temp_image_path)
                
                if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                    class_name, confidence = prediction_result
                    
                    # Handle numeric class prediction
                    if isinstance(class_name, (int, np.integer)):
                        class_name = class_names.get(int(class_name), f"Unknown Class {class_name}")
                    
                    self.ai_prediction = class_name
                    self.ai_confidence = float(confidence)
                    
                    # Update AI results display
                    self.update_ai_display(class_name, confidence)
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_image_path)
                    except:
                        pass
                    
                else:
                    raise ValueError("Invalid prediction format returned from AI model")
                    
            except Exception as ai_error:
                print(f"AI prediction error: {ai_error}")
                self.ai_result_label.setText("AI Analysis: Model Error")
                self.ai_result_label.setStyleSheet("""
                    font-size: 14px;
                    font-weight: bold;
                    color: #e74c3c;
                    padding: 10px;
                    border: 1px solid #e74c3c;
                    border-radius: 5px;
                    background-color: #fdf2f2;
                """)
                self.confidence_label.setText("AI model unavailable")
                self.ai_prediction = "Analysis Error"
                self.ai_confidence = 0.0
            
            # Enable export buttons
            self.save_btn.setEnabled(True)
            self.save_btn.setStyleSheet(self.get_button_style())
            self.save_contour_btn.setEnabled(True)
            self.save_contour_btn.setStyleSheet(self.get_button_style())
            
            # Show completion message
            contour_count = len(contours) if contours else 0
            QMessageBox.information(self, "Analysis Complete", 
                                  f"Analysis completed successfully!\n\n"
                                  f"Computer Vision Results:\n"
                                  f"• Detected {contour_count} contour region(s)\n"
                                  f"• Threshold: {self.threshvalue}\n"
                                  f"• Line density: {self.linestep}\n\n"
                                  f"AI Classification: {self.ai_prediction}\n"
                                  f"Confidence: {self.ai_confidence*100:.1f}%\n\n"
                                  f"Results are ready for export.")
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze injury: {str(e)}")
            print(f"Analysis error: {e}")
            traceback.print_exc()
            self.result_label.setText("Analysis failed.\nCheck console for details.")

    def update_ai_display(self, class_name, confidence):
        """Update the AI results display with classification results"""
        # Determine confidence color based on value
        if confidence >= 0.8:
            confidence_color = "#27ae60"  # Green - high confidence
            bg_color = "#d5f4e6"
            border_color = "#27ae60"
        elif confidence >= 0.6:
            confidence_color = "#f39c12"  # Orange - medium confidence  
            bg_color = "#fef9e7"
            border_color = "#f39c12"
        else:
            confidence_color = "#e74c3c"  # Red - low confidence
            bg_color = "#fdf2f2"
            border_color = "#e74c3c"
        
        # Update prediction label
        self.ai_result_label.setText(f"AI Diagnosis: {class_name}")
        self.ai_result_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {confidence_color};
            padding: 10px;
            border: 2px solid {border_color};
            border-radius: 5px;
            background-color: {bg_color};
        """)
        
        # Update confidence label
        self.confidence_label.setText(f"Confidence: {confidence*100:.1f}%")
        self.confidence_label.setStyleSheet(f"""
            font-size: 12px;
            font-weight: bold;
            color: {confidence_color};
            padding: 5px;
        """)

    def save_result(self):
        """Save comprehensive analysis results with AI diagnosis"""
        if self.result_img is None:
            return
            
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Analysis Results - RoboDoc v2", "robodoc_v2_analysis.png", 
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
            )
            if path:
                # Create annotated result image with AI diagnosis
                annotated_result = self.create_annotated_result()
                
                success = cv2.imwrite(path, annotated_result)
                if success:
                    QMessageBox.information(self, "Export Successful", 
                                          f"Complete analysis results saved successfully!\n\n"
                                          f"Location: {path}\n\n"
                                          f"Includes:\n"
                                          f"• Computer vision analysis\n"
                                          f"• AI diagnosis: {self.ai_prediction}\n"
                                          f"• Confidence: {self.ai_confidence*100:.1f}%")
                else:
                    QMessageBox.warning(self, "Export Error", "Failed to save the analysis results.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save results: {str(e)}")

    def create_annotated_result(self):
        """Create annotated result image with AI diagnosis overlay"""
        try:
            # Create a copy of the result image
            annotated = self.result_img.copy()
            
            # Add AI diagnosis overlay
            if self.ai_prediction and self.ai_confidence > 0:
                # Create text overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                
                # Prepare text
                diagnosis_text = f"AI Diagnosis: {self.ai_prediction}"
                confidence_text = f"Confidence: {self.ai_confidence*100:.1f}%"
                
                # Calculate text sizes
                (text_w1, text_h1), _ = cv2.getTextSize(diagnosis_text, font, font_scale, thickness)
                (text_w2, text_h2), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
                
                # Create overlay rectangle
                overlay_height = text_h1 + text_h2 + 40
                overlay_width = max(text_w1, text_w2) + 40
                
                # Position at top of image
                x_offset = 10
                y_offset = 10
                
                # Draw semi-transparent background
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x_offset, y_offset), 
                            (x_offset + overlay_width, y_offset + overlay_height), 
                            (0, 0, 0), -1)  # Black background
                
                # Blend with original
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
                
                # Add text
                cv2.putText(annotated, diagnosis_text, 
                          (x_offset + 20, y_offset + text_h1 + 20), 
                          font, font_scale, (0, 255, 255), thickness)  # Yellow text
                
                cv2.putText(annotated, confidence_text, 
                          (x_offset + 20, y_offset + text_h1 + text_h2 + 30), 
                          font, font_scale, (0, 255, 255), thickness)  # Yellow text
            
            return annotated
            
        except Exception as e:
            print(f"Error creating annotated result: {e}")
            return self.result_img

    def save_selected_contours(self):
        """Save isolated contours with AI diagnosis metadata"""
        if self.injury_img is None or self.contours is None:
            return
            
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Contours - RoboDoc v2", "robodoc_v2_contours.png", 
                "PNG Files (*.png);;All Files (*)"
            )
            if path:
                save_selected_contours(self.injury_img, self.contours, path)
                
                # Also save metadata file
                metadata_path = path.replace('.png', '_metadata.txt')
                with open(metadata_path, 'w') as f:
                    f.write(f"RoboDoc v2 - Contour Analysis Metadata\n")
                    f.write(f"=====================================\n\n")
                    f.write(f"AI Diagnosis: {self.ai_prediction or 'N/A'}\n")
                    f.write(f"Confidence: {self.ai_confidence*100:.2f}%\n")
                    f.write(f"Contour Count: {len(self.contours)}\n")
                    f.write(f"Detection Threshold: {self.threshvalue}\n")
                    f.write(f"Line Density: {self.linestep}\n")
                    f.write(f"Original Image Dimensions: {self.injury_img.shape}\n")
                
                QMessageBox.information(self, "Export Successful", 
                                      f"Contours and metadata exported successfully!\n\n"
                                      f"Files saved:\n"
                                      f"• {path}\n"
                                      f"• {metadata_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export contours: {str(e)}")

    def reset_analysis_state(self):
        """Reset the analysis state when new image is loaded"""
        self.cropped_control = None
        self.result_img = None
        self.contours = None
        self.ai_prediction = None
        self.ai_confidence = 0.0
        
        # Reset UI elements
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet(self.get_button_style(disabled=True, primary=True))
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(self.get_button_style(disabled=True))
        self.save_contour_btn.setEnabled(False)
        self.save_contour_btn.setStyleSheet(self.get_button_style(disabled=True))
        
        # Reset AI display
        self.ai_result_label.setText("AI Prediction: Awaiting Analysis")
        self.ai_result_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            background-color: #ecf0f1;
        """)
        self.confidence_label.setText("Confidence: ---%")
        self.confidence_label.setStyleSheet("""
            font-size: 12px;
            color: #7f8c8d;
            padding: 5px;
        """)

    def update_thresh(self, val):
        self.threshvalue = val
        self.thresh_value_label.setText(str(val))

    def update_line(self, val):
        self.linestep = val
        self.line_value_label.setText(str(val))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontShowIconsInMenus, False)
    
    app.setApplicationName("RoboDoc v2")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("MSA University - Medical Imaging Solutions")
    
    try:
        window = RoboDocV2App()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()