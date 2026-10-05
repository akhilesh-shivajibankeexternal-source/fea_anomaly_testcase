from ultralytics import YOLO

def train_custom_yolo():
    print("Loading local YOLOv26 Small model...")
    model = YOLO("yolo26s.pt") 

    print("Starting YOLO Training...")
    results = model.train(
        data="data.yaml",   
        epochs=50,          
        imgsz=640,          
        batch=8,            
        device="cpu", # "cuda" if using a GPU
        plots=True          
    )
    
if __name__ == '__main__':
    train_custom_yolo()