    
import cv2
import numpy as np
import torch


class Visualiser:
    def __init__(self):
        pass

    def get_overlay(self, image, mask):
        COLORS = ((0, 0, 0), (0, 128, 255), (255, 0, 0))
        color_mask = np.array(COLORS)[mask]

        overlay = np.zeros(image.shape[1:], dtype=np.uint8)
        overlay = (0.5 * image) + (0.5 * color_mask)

        overlay[mask==0] = image[mask==0]
        overlay[mask==2] = color_mask[mask==2]

        overlay = overlay.astype("uint8")
        return overlay
    
    def _create_image_grid(self, images, grid_shape, border_color=(255, 255, 255), border_thickness=2):
        rows, cols = grid_shape
        grid_height = max(img.shape[0] for img in images) * rows
        grid_width = max(img.shape[1] for img in images) * cols
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            y_start = row * img.shape[0]
            y_end = y_start + img.shape[0]
            x_start = col * img.shape[1]
            x_end = x_start + img.shape[1]
            grid[y_start:y_end, x_start:x_end] = img
            cv2.rectangle(grid, (x_start, y_start), (x_end, y_end), border_color, border_thickness)
        
        return grid
    
    def create_grid(self, image, data, names, frame_size=(1280, 720), show_names=True):
        img = cv2.resize(image, frame_size)

        for i in range(3):
            if i < len(data):
                data[i] = data[i].detach().cpu().long().numpy()
            else:
                data.append(torch.zeros_like(data[i-1].shape))

        overlay1 = cv2.cvtColor(
            self.get_overlay(img, cv2.resize(data[0].astype(np.uint8), frame_size)), 
            cv2.COLOR_RGB2BGR
        )
        overlay2 = cv2.cvtColor(
            self.get_overlay(img, cv2.resize(data[1].astype(np.uint8), frame_size)), 
            cv2.COLOR_RGB2BGR
        )
        overlay3 = cv2.cvtColor(
            self.get_overlay(img, cv2.resize(data[2].astype(np.uint8), frame_size)), 
            cv2.COLOR_RGB2BGR
        )

        if show_names:
            greybox = (70, 400)
            overlay1[:greybox[0], :greybox[1], :] = (overlay1[:greybox[0], :greybox[1], :].astype(np.float32) *0.4).astype(np.uint8)
            overlay1 = cv2.putText(
                overlay1, 
                names[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA
            )
            overlay2[:greybox[0], :greybox[1], :] = (overlay2[:greybox[0], :greybox[1], :].astype(np.float32) *0.4).astype(np.uint8)
            overlay2 = cv2.putText(
                overlay2, 
                names[1], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA
            )
            overlay3[:greybox[0], :greybox[1], :] = (overlay3[:greybox[0], :greybox[1], :].astype(np.float32) *0.4).astype(np.uint8)
            overlay3 = cv2.putText(
                overlay3, 
                names[2], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA
            )

        grid_image = self._create_image_grid([
                cv2.cvtColor(cv2.resize(image.astype(np.uint8), frame_size), cv2.COLOR_BGR2RGB),
                cv2.resize(overlay1.astype(np.uint8), frame_size),
                cv2.resize(overlay2.astype(np.uint8), frame_size),
                cv2.resize(overlay3.astype(np.uint8), frame_size),
            ], (2, 2)
        )

        return grid_image
