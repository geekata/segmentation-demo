import os
import queue
import threading
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from matplotlib import colormaps
from sam_model import SAMSegmenter
from tkinter import filedialog, messagebox, Canvas
from last_capture_request import get_last_capture, download_capture

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class SegmentationDemoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Segmentation Demo")
        self.geometry("1180x760")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.current_image = None
        self.original_image = None
        self.max_display_size = (1600, 900)
        self.segmenter = SAMSegmenter()
        self.drawing_box = False
        self.image_box = None
        self.box_start = None
        self.box_input = None
        self.point_inputs = []
        self.task_queue = queue.Queue()
        self.thread_running = False
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # Left Panel
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nswe")
        self.sidebar.grid_rowconfigure((0, 1, 2, 3), minsize=20)

        # "Load Image" title
        self.open_label = ctk.CTkLabel(self.sidebar, text="Load Image", font=ctk.CTkFont(size=18))
        self.open_label.grid(row=0, column=0, padx=20, pady=(10, 10), sticky="w")

        # "Last Capture" button
        self.last_capture_button = ctk.CTkButton(self.sidebar, text="Last Capture", command=self.load_last_capture)
        self.last_capture_button.grid(row=1, column=0, padx=20, pady=5, sticky="we")

        # "Select File" button
        self.upload_button = ctk.CTkButton(self.sidebar, text="Select File", command=self.upload_image)
        self.upload_button.grid(row=2, column=0, padx=20, pady=5, sticky="we")

        # "Mode" title
        self.mode_label = ctk.CTkLabel(self.sidebar, text="Mode", font=ctk.CTkFont(size=18))
        self.mode_label.grid(row=3, column=0, padx=20, pady=(35, 10), sticky="w")

        # Mode options
        self.mode = ctk.StringVar(value="Everything")
        self.mode_dropdown = ctk.CTkOptionMenu(self.sidebar, values=["Everything", "Box", "Point & Click"], variable=self.mode)
        self.mode.trace_add("write", lambda *args: self.clear_all())
        self.mode_dropdown.grid(row=4, column=0, padx=20, pady=5, sticky="we")

        # Spacer row to push buttons to the bottom
        self.sidebar.grid_rowconfigure(5, weight=1)

        # "Run" Button
        self.run_button = ctk.CTkButton(self.sidebar, text="Run", command=self.run_segmentation, state="disabled")
        self.run_button.grid(row=6, column=0, padx=20, pady=5, sticky="we")

        # "Clear" Button
        self.clear_button = ctk.CTkButton(self.sidebar, text="Clear", command=self.clear_all, state="disabled")
        self.clear_button.grid(row=7, column=0, padx=20, pady=(5, 10), sticky="we")

        # Right Panel: Image Viewer
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Canvas for drawing input
        self.canvas = Canvas(self.image_frame, bg="#212121", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        # self.no_image_text = self.canvas.create_text(
        #     self.canvas.winfo_width() // 2,
        #     self.canvas.winfo_height() // 2,
        #     text="No image loaded",
        #     fill="white",
        #     font=("Arial", 16),
        #     anchor="center"
        # )

        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def load_last_capture(self):
        if self.thread_running:
            return
        self.task_queue.put({"type": "download"})

    def upload_image(self):
        if self.thread_running:
            return
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.task_queue.put({"type": "upload", "path": path})

    def run_segmentation(self, *args):
        if self.original_image is None or self.thread_running:
            print("No image loaded.")
            return
        self.task_queue.put({"type": "segment", "mode": self.mode.get()})

    def clear_inputs(self):
        self.box_start = None
        self.box_input = None
        self.point_inputs = []
        if self.mode.get() == "Everything":
            self.run_button.configure(state="normal")
        else:
            self.run_button.configure(state="disabled")

    def clear_all(self):
        self.clear_inputs()
        if self.original_image is None:
            return
        self.current_image = ImageTk.PhotoImage(self.original_image)
        self._display_image(self.current_image)

    def reset_buttons(self, enable=True):
        buttons = [self.last_capture_button, self.upload_button, self.mode_dropdown, self.run_button, self.clear_button]
        if self.thread_running:
            for button in buttons:
                button.configure(state="disabled")
        else:
            for button in buttons:
                button.configure(state="normal")

    def canvas_to_image_coords(self, x, y):
        start_x, start_y = self.image_box[0], self.image_box[1]
        return x - start_x, y - start_y

    def get_box(self):
        box_coords = self.canvas.coords(self.box_input)
        x0, y0 = self.canvas_to_image_coords(box_coords[0], box_coords[1])
        x1, y1 = self.canvas_to_image_coords(box_coords[2], box_coords[3])
        x_min = min(x0, x1)
        y_min = min(y0, y1)
        x_max = max(x0, x1)
        y_max = max(y0, y1)
        return np.array([x_min, y_min, x_max, y_max])

    def get_points(self):
        point_coords = []
        for point in self.point_inputs:
            x, y = self.canvas_to_image_coords(*point)
            point_coords.append([x, y])
        return np.array(point_coords)

    def on_canvas_resize(self, event):
        if self.current_image is None:
            return
        self.clear_inputs()
        self._display_image(self.current_image)

    def on_mouse_down(self, event):
        if self.current_image is None:
            return
        if not self._is_within_image(event.x, event.y):
            return

        mode = self.mode.get()
        if mode == "Box":
            self.drawing_box = True
            self.canvas.delete("input_overlay")
            self.box_start = (event.x, event.y)
            self.box_input = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                          outline="red", width=5, tags="input_overlay")
        elif mode == "Point & Click":
            self.point_inputs.append((event.x, event.y))
            self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                                    fill="red", outline="white", width=1, tags="input_overlay")
            self.run_button.configure(state="normal")

    def on_mouse_drag(self, event):
        if self.drawing_box and self.box_input:
            x2 = min(max(event.x, self.image_box[0]), self.image_box[2])
            y2 = min(max(event.y, self.image_box[1]), self.image_box[3])
            self.canvas.coords(self.box_input, *self.box_start, x2, y2)

    def on_mouse_up(self, event):
        if self.current_image is None or not self.drawing_box:
            return
        self.drawing_box = False

        x2 = min(max(event.x, self.image_box[0]), self.image_box[2])
        y2 = min(max(event.y, self.image_box[1]), self.image_box[3])
        self.canvas.coords(self.box_input, *self.box_start, x2, y2)
        self.run_button.configure(state="normal")

    def _worker_loop(self):
        while True:
            task = self.task_queue.get()
            if task["type"] == "download":
                self._handle_download()
            elif task["type"] == "upload":
                self._handle_upload(task["path"])
            elif task["type"] == "segment":
                self._handle_segmentation(task["mode"])

            self.task_queue.task_done()

    def _display_image(self, image):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        image_w = image.width()
        image_h = image.height()

        # Calculate top-left coordinates to center the image
        x0 = (canvas_w - image_w) // 2
        y0 = (canvas_h - image_h) // 2
        self.image_box = (x0, y0, x0 + image_w, y0 + image_h)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, anchor="center", image=image)
        self.canvas.image = image

    def _handle_download(self):
        try:
            self.thread_running = True
            self.reset_buttons()
            newest = get_last_capture()
            if newest:
                path = download_capture(newest)
                print(path)
                self.task_queue.put({"type": "upload", "path": path})

        except Exception as e:
            self.thread_running = False
            self.after(0, lambda: [
                self.reset_buttons(),
                messagebox.showerror("Error", str(e))
            ])

        finally:
            self.thread_running = False
            self.after(0, lambda: self.reset_buttons())

    def _handle_upload(self, path):
        try:
            self.thread_running = True
            self.reset_buttons()

            img = Image.open(path).convert("RGB")
            img.thumbnail(self.max_display_size, Image.Resampling.LANCZOS)
            self.original_image = img
            self.current_image = ImageTk.PhotoImage(img)
            self.segmenter.set_image_array(np.array(img))

        except Exception as e:
            self.thread_running = False
            self.after(0, lambda: [
                self.canvas.delete("all"),
                self.reset_buttons(),
                messagebox.showerror("Error", str(e))
            ])
        finally:
            self.thread_running = False
            self.after(0, lambda: [
                self._display_image(self.current_image),
                self.reset_buttons(),
                self.clear_inputs()
            ])

    def _handle_segmentation(self, mode):
        try:
            if self.original_image is None:
                print("No image loaded.")
                return

            self.thread_running = True
            self.reset_buttons()

            img = np.array(self.original_image.convert("RGB"))
            self.segmenter.set_image_array(img)

            h, w = img.shape[:2]
            if mode == "Everything":
                box = np.array([0, 0, w - 1, h - 1])
                masks = self.segmenter.segment_with_box(box)
            elif mode == "Box" and self.box_input:
                box = self.get_box()
                masks = self.segmenter.segment_with_box(box)
            elif mode == "Point & Click" and self.point_inputs:
                points = self.get_points()
                masks = self.segmenter.segment_with_point(points)
            else:
                print("Unsupported mode.")
                self.thread_running = False
                return None

            # Composite mask
            base_img = img.astype(np.float32).copy()
            if masks.ndim == 2:
                masks = np.expand_dims(masks, axis=0)

            color_map = colormaps["hsv"]
            colors = [np.array(color_map(i / masks.shape[0])[:3]) * 255 for i in range(masks.shape[0])]
            for i, mask in enumerate(masks):
                base_img[mask] = 0.6 * base_img[mask] + 0.4 * colors[i]

            result_image = Image.fromarray(base_img.astype(np.uint8))
            result_image.thumbnail(self.max_display_size, Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(result_image)

        except Exception as e:
            print("Segmentation error:", e)
            self.thread_running = False,
            self.after(0, lambda: [
                self.reset_buttons(),
                messagebox.showerror("Error", str(e))
            ])

        finally:
            self.thread_running = False
            self.after(0, lambda: [
                self._display_image(self.current_image),
                self.reset_buttons(),
                self.clear_inputs()
            ])

    def _is_within_image(self, x, y):
        if not self.image_box:
            return False
        x0, y0, x1, y1 = self.image_box
        return x0 <= x <= x1 and y0 <= y <= y1


if __name__ == "__main__":
    app = SegmentationDemoApp()
    app.mainloop()
