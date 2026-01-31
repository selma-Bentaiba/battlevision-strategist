import numpy as np
import cv2
from PIL import Image

def detect_objects(image):
  
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Simple edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and create detections
    detections = []
    min_area = 1000
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Simulate confidence score
            confidence = min(0.95, area / 10000)
            
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': confidence,
                'class': 'object',
                'id': i
            })
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:10]
    
    # Create annotated image
    annotated = image.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        label = f"{det['class']}: {det['confidence']:.2f}"
        cv2.putText(annotated, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return detections, annotated


def apply_patch(image, patch_type, placement, intensity=0.7):
    """
    Apply adversarial patch to image
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    # Determine patch size (10-20% of image)
    patch_size = int(min(h, w) * 0.15)
    
    # Determine placement location
    if placement == "Center":
        x, y = w//2 - patch_size//2, h//2 - patch_size//2
    elif placement == "Top-Left":
        x, y = 10, 10
    elif placement == "Bottom-Right":
        x, y = w - patch_size - 10, h - patch_size - 10
    else:  # Random
        x = np.random.randint(0, max(1, w - patch_size))
        y = np.random.randint(0, max(1, h - patch_size))
    
    # Create patch based on type
    if patch_type == "Camouflage Pattern":
        # Create camouflage-like pattern
        patch = create_camouflage_patch(patch_size, intensity)
    elif patch_type == "Geometric Shapes":
        # Create geometric pattern
        patch = create_geometric_patch(patch_size, intensity)
    elif patch_type == "Texture Noise":
        # Create texture noise
        patch = create_texture_patch(patch_size, intensity)
    else:  # Random Pixels
        # Random noise
        patch = np.random.randint(0, 256, (patch_size, patch_size, 3)).astype(np.uint8)
    
    # Apply patch to image
    x_end = min(x + patch_size, w)
    y_end = min(y + patch_size, h)
    patch_h = y_end - y
    patch_w = x_end - x
    
    # Blend patch with image
    alpha = intensity
    result[y:y_end, x:x_end] = (alpha * patch[:patch_h, :patch_w] + 
                                 (1 - alpha) * result[y:y_end, x:x_end]).astype(np.uint8)
    
    return result


def create_camouflage_patch(size, intensity):
    """Create camouflage-style patch"""
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Multiple colored blobs
    colors = [
        [34, 139, 34],   # Forest green
        [101, 67, 33],   # Brown
        [139, 90, 43],   # Tan
        [85, 107, 47]    # Dark olive
    ]
    
    for _ in range(8):
        color = colors[np.random.randint(0, len(colors))]
        center = (np.random.randint(0, size), np.random.randint(0, size))
        radius = np.random.randint(size//8, size//3)
        cv2.circle(patch, center, radius, color, -1)
    
    # Add some noise
    noise = np.random.randint(-30, 30, (size, size, 3), dtype=np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Blur to blend
    patch = cv2.GaussianBlur(patch, (15, 15), 0)
    
    return patch


def create_geometric_patch(size, intensity):
    """Create geometric pattern patch"""
    patch = np.ones((size, size, 3), dtype=np.uint8) * 128
    
    # Create stripes
    stripe_width = size // 10
    for i in range(0, size, stripe_width * 2):
        patch[i:i+stripe_width, :] = [200, 50, 50]
    
    # Add circles
    for _ in range(5):
        center = (np.random.randint(0, size), np.random.randint(0, size))
        radius = np.random.randint(size//15, size//6)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.circle(patch, center, radius, color, -1)
    
    # Add rectangles
    for _ in range(3):
        pt1 = (np.random.randint(0, size), np.random.randint(0, size))
        pt2 = (np.random.randint(0, size), np.random.randint(0, size))
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.rectangle(patch, pt1, pt2, color, -1)
    
    return patch


def create_texture_patch(size, intensity):
    """Create texture noise patch"""
    # Generate Perlin-like noise
    patch = np.random.randint(0, 256, (size//4, size//4, 3), dtype=np.uint8)
    
    # Upscale with interpolation
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC)
    
    # Add high-frequency noise
    noise = np.random.randint(-50, 50, (size, size, 3), dtype=np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return patch


def defend_image(image, defense_type, strength=0.6):
    """
    Apply defense mechanism to image
    """
    result = image.copy()
    
    # Scale strength to appropriate range for each filter
    kernel_size = int(3 + strength * 8)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd
    
    if defense_type == "Gaussian Denoising":
        # Apply Gaussian blur
        sigma = strength * 3
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), sigma)
        
    elif defense_type == "Median Filter":
        # Apply median filter
        result = cv2.medianBlur(result, kernel_size)
        
    elif defense_type == "Bilateral Filter":
        # Apply bilateral filter (preserves edges)
        d = kernel_size
        sigma_color = strength * 100
        sigma_space = strength * 100
        result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
        
    elif defense_type == "JPEG Compression":
        # Simulate JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(100 - strength * 70)]
        _, encoded = cv2.imencode('.jpg', result, encode_param)
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result


def visualize_detection(image, detections):
    """
    Create visualization of detections
    """
    annotated = image.copy()
    
    for det in detections:
        x, y, w, h = det['bbox']
        confidence = det['confidence']
        
        # Color based on confidence
        color = (
            int(255 * (1 - confidence)),  # R
            int(255 * confidence),         # G
            0                              # B
        )
        
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        
        label = f"{det['class']}: {confidence:.2f}"
        cv2.putText(annotated, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated


def generate_attention_heatmap(image, detections):
    """
    Generate attention heatmap showing where model focuses
    """
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for det in detections:
        x, y, bbox_w, bbox_h = det['bbox']
        confidence = det['confidence']
        
        # Create Gaussian around detection
        y_coords, x_coords = np.ogrid[:h, :w]
        center_y, center_x = y + bbox_h//2, x + bbox_w//2
        
        sigma = max(bbox_w, bbox_h) / 2
        gaussian = np.exp(-((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * sigma**2))
        
        heatmap += gaussian * confidence
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Convert to color
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original
    result = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    return result
