from ultralytics import YOLO

model = YOLO('best.pt')

source = 'Screen Recording 2024-04-12 at 6.12.51 PM.mov'

results = model(source,save=True)