import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import tensorflow as tf

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện chữ số viết tay")
        self.root.geometry("600x500")
        
        # Load mô hình đã train
        try:
            self.model = tf.keras.models.load_model("model.h5")
            print("Đã load mô hình thành công!")
        except:
            messagebox.showerror("Lỗi", "Không thể load mô hình model.h5. Hãy đảm bảo file tồn tại!")
            return
        
        # Khởi tạo canvas để vẽ (28x28 pixels, scale lên 10 lần để dễ vẽ)
        self.canvas_size = 280  # 28 * 10
        self.scale_factor = 10
        
        # Tạo image để lưu trữ nội dung vẽ
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)  # Nền trắng
        self.draw = ImageDraw.Draw(self.image)
        
        self.setup_ui()
        
        # Biến để theo dõi trạng thái vẽ
        self.old_x = None
        self.old_y = None
        
    def setup_ui(self):
        # Tiêu đề
        title_label = tk.Label(self.root, text="Nhận diện chữ số viết tay", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Frame chính chứa canvas và kết quả
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=10)
        
        # Frame cho canvas
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, padx=20)
        
        canvas_label = tk.Label(canvas_frame, text="Vẽ số ở đây (28x28 px):", 
                               font=("Arial", 12))
        canvas_label.pack()
        
        # Canvas để vẽ
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, 
                               height=self.canvas_size, bg='white', 
                               relief=tk.SUNKEN, border=2)
        self.canvas.pack(pady=5)
        
        # Bind sự kiện vẽ
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonPress-1>', self.start_paint)
        self.canvas.bind('<ButtonRelease-1>', self.end_paint)
        
        # Frame cho kết quả dự đoán
        result_frame = tk.Frame(main_frame)
        result_frame.pack(side=tk.LEFT, padx=20)
        
        result_title = tk.Label(result_frame, text="Kết quả dự đoán:", 
                               font=("Arial", 12, "bold"))
        result_title.pack()
        
        # Label hiển thị số dự đoán
        self.prediction_label = tk.Label(result_frame, text="?", 
                                        font=("Arial", 48, "bold"), 
                                        fg="blue", width=3, height=2,
                                        relief=tk.RIDGE, border=2)
        self.prediction_label.pack(pady=10)
        
        # Label hiển thị độ tin cậy
        self.confidence_label = tk.Label(result_frame, text="Độ tin cậy: --", 
                                        font=("Arial", 10))
        self.confidence_label.pack()
        
        # Frame cho các nút điều khiển
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Nút dự đoán
        predict_btn = tk.Button(button_frame, text="Dự đoán", 
                               command=self.predict_drawn_digit,
                               font=("Arial", 12), bg="lightgreen", 
                               width=12, height=2)
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Nút xóa canvas
        clear_btn = tk.Button(button_frame, text="Xóa", 
                             command=self.clear_canvas,
                             font=("Arial", 12), bg="lightcoral", 
                             width=12, height=2)
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Nút tải ảnh
        upload_btn = tk.Button(button_frame, text="Tải ảnh", 
                              command=self.upload_image,
                              font=("Arial", 12), bg="lightblue", 
                              width=12, height=2)
        upload_btn.pack(side=tk.LEFT, padx=10)
        
    def start_paint(self, event):
        self.old_x = event.x
        self.old_y = event.y
        
    def paint(self, event):
        if self.old_x and self.old_y:
            # Vẽ trên canvas
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                   width=15, fill='black', capstyle=tk.ROUND, 
                                   smooth=tk.TRUE)
            
            # Vẽ trên PIL Image (để lưu trữ)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                          fill=0, width=15)
            
        self.old_x = event.x
        self.old_y = event.y
        
    def end_paint(self, event):
        self.old_x = None
        self.old_y = None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        # Tạo lại image trắng
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        # Reset kết quả
        self.prediction_label.config(text="?")
        self.confidence_label.config(text="Độ tin cậy: --")
        
    def preprocess_image(self, img):
        """Tiền xử lý ảnh để phù hợp với mô hình"""
        # Resize về 28x28
        img_resized = img.resize((28, 28), Image.LANCZOS)
        
        # Chuyển thành array numpy
        img_array = np.array(img_resized)
        
        # Đảo ngược màu (nền đen, chữ trắng) như MNIST
        img_array = 255 - img_array
        
        # Normalize về [0,1]
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape để phù hợp với input của mô hình
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
        
    def predict_drawn_digit(self):
        """Dự đoán chữ số được vẽ trên canvas"""
        try:
            # Preprocess ảnh
            processed_img = self.preprocess_image(self.image)
            
            # Dự đoán
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # Hiển thị kết quả
            self.prediction_label.config(text=str(predicted_digit))
            self.confidence_label.config(text=f"Độ tin cậy: {confidence:.1f}%")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán: {str(e)}")
            
    def upload_image(self):
        """Tải ảnh từ file và dự đoán"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh chứa chữ số",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Load ảnh
                uploaded_img = Image.open(file_path)
                
                # Chuyển thành grayscale nếu cần
                if uploaded_img.mode != 'L':
                    uploaded_img = uploaded_img.convert('L')
                
                # Preprocess và dự đoán
                processed_img = self.preprocess_image(uploaded_img)
                predictions = self.model.predict(processed_img, verbose=0)
                predicted_digit = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                # Hiển thị kết quả
                self.prediction_label.config(text=str(predicted_digit))
                self.confidence_label.config(text=f"Độ tin cậy: {confidence:.1f}%")
                
                # Hiển thị ảnh đã upload trên canvas (tùy chọn)
                display_img = uploaded_img.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)
                self.photo = ImageTk.PhotoImage(display_img)
                self.canvas.delete("all")
                self.canvas.create_image(self.canvas_size//2, self.canvas_size//2, 
                                       image=self.photo)
                
                # Cập nhật image để vẽ tiếp
                self.image = uploaded_img.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)
                self.draw = ImageDraw.Draw(self.image)
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể load ảnh: {str(e)}")

def main():
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()