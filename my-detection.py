import jetson.inference
import jetson.utils

# 加载预训练模型
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# 改用图像作为输入
image_path = "0c0f47e6e9354f51.jpg"  # 替换为你实际的图像路径
img = jetson.utils.loadImage(image_path)

# 执行检测
detections = net.Detect(img)

# 打印所有检测结果
print("All Detections:")
print(detections)  # 显示原始检测对象列表

# 遍历检测结果并打印详细信息
print("\nDetailed Detection Results:")
for detection in detections:
    class_name = net.GetClassDesc(detection.ClassID)
    print(f"Class: {class_name}, Coordinates: ({detection.Left}, {detection.Top}, {detection.Right}, {detection.Bottom}), "
          f"Confidence: {detection.Confidence:.2f}")

# 显示检测结果
output_path = "output_image.jpg"  # 结果保存路径
jetson.utils.saveImage(output_path, img)
print(f"\nDetection results saved to {output_path}")
