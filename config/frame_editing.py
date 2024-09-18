import cv2


class FrameEditing:
    @staticmethod
    def convert_frame_to_gray(frame):
        # Check if frame is valid
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        # Convert frame to grayscale
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            raise ValueError(f"Error converting frame to grayscale: {e}")

    @staticmethod
    def convert_frame_color_space(frame, color_space='gray'):
        """ Convert frame to different color spaces.
        Args:
            frame: Input frame to be converted.
            color_space: The color space to convert to, options are 'gray', 'hsv', 'lab'.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        color_spaces = {
            'gray': cv2.COLOR_BGR2GRAY,
            'hsv': cv2.COLOR_BGR2HSV,
            'lab': cv2.COLOR_BGR2LAB
        }

        if color_space not in color_spaces:
            raise ValueError(f"Invalid color space: {color_space}. Available options: {list(color_spaces.keys())}")

        try:
            return cv2.cvtColor(frame, color_spaces[color_space])
        except cv2.error as e:
            raise ValueError(f"Error converting frame to {color_space}: {e}")

    @staticmethod
    def scale_frame(frame, fx=None, fy=None, width=None, height=None, keep_aspect_ratio=False,
                    interpolation=cv2.INTER_LINEAR):
        """
        Resizes the frame based on scaling factors or desired dimensions.
        Args:
            frame: Input frame to be resized.
            fx: Scaling factor along the x-axis (horizontal).
            fy: Scaling factor along the y-axis (vertical).
            width: The desired output width (used if fx and fy are not provided).
            height: The desired output height (used if fx and fy are not provided).
            keep_aspect_ratio: Whether to preserve the aspect ratio when resizing.
            interpolation: Interpolation method for resizing, default is INTER_LINEAR.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        h, w = frame.shape[:2]

        # Preserve aspect ratio if specified
        if keep_aspect_ratio and width and height:
            raise ValueError("Cannot specify both width/height and keep_aspect_ratio=True.")

        if keep_aspect_ratio:
            if width:
                aspect_ratio = w / h
                height = int(width / aspect_ratio)
            elif height:
                aspect_ratio = h / w
                width = int(height * aspect_ratio)

        # Use absolute dimensions if provided, otherwise use scaling factors
        if width and height:
            return cv2.resize(frame, (width, height), interpolation=interpolation)
        elif fx is not None and fy is not None:
            return cv2.resize(frame, (0, 0), fx=fx, fy=fy, interpolation=interpolation)
        else:
            raise ValueError("Either fx and fy or width and height must be provided for resizing.")
