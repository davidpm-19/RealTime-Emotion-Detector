import cv2


class FrameDrawing:
    @staticmethod
    def draw_rect(frame, pt1, pt2, color=(0, 255, 0), thickness=2, alpha=1.0):
        """
        Draws a rectangle on the frame.

        Args:
            frame: Input frame on which the rectangle is drawn.
            pt1: Top-left corner of the rectangle (x, y).
            pt2: Bottom-right corner of the rectangle (x, y).
            color: Color of the rectangle as a BGR tuple. Default is green (0, 255, 0).
            thickness: Thickness of the rectangle's border. Default is 2.
            alpha: Transparency level of the rectangle. Default is 1.0 (fully opaque).
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        overlay = frame.copy()
        cv2.rectangle(overlay, pt1, pt2, color, thickness)

        # Apply transparency blending
        if 0 <= alpha < 1:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            frame[:] = overlay[:]

    @staticmethod
    def draw_text(frame, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2,
                  alpha=1.0):
        """
        Draws text on the frame.

        Args:
            frame: Input frame on which the text is drawn.
            text: The text string to draw.
            org: Bottom-left corner of the text in the image (x, y).
            font: Font type. Default is FONT_HERSHEY_SIMPLEX.
            font_scale: Font scale factor that is multiplied by the font-specific base size.
            color: Color of the text as a BGR tuple. Default is white (255, 255, 255).
            thickness: Thickness of the text stroke. Default is 2.
            alpha: Transparency level of the text. Default is 1.0 (fully opaque).
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        overlay = frame.copy()
        cv2.putText(overlay, text, org, font, font_scale, color, thickness)

        # Apply transparency blending
        if 0 <= alpha < 1:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            frame[:] = overlay[:]

    @staticmethod
    def draw_circle(frame, center, radius, color=(0, 0, 255), thickness=2, alpha=1.0):
        """
        Draws a circle on the frame.

        Args:
            frame: Input frame on which the circle is drawn.
            center: Center of the circle (x, y).
            radius: Radius of the circle.
            color: Color of the circle as a BGR tuple. Default is red (0, 0, 255).
            thickness: Thickness of the circle's border. Default is 2.
            alpha: Transparency level of the circle. Default is 1.0 (fully opaque).
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        overlay = frame.copy()
        cv2.circle(overlay, center, radius, color, thickness)

        # Apply transparency blending
        if 0 <= alpha < 1:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            frame[:] = overlay[:]
