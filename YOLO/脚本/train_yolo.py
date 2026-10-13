
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

dataset_dir = Path(r"D:\工具\YOLO\data\dataset")


# 1. 数据预处理(实际训练数据集是早划分完的,这里的代码是为了让提交文件完整后面补充的)
def prepare_data():
    raw_dir = Path(r"D:\工具\YOLO\data\raw")
    output_dir = Path(r"D:\工具\YOLO\data\dataset_new")

    # 读取原始图片
    images = sorted((raw_dir / "images").glob("*.jpg"))

    if not images:
        raise ValueError("未找到原始图片")

    if output_dir.exists():
        raise FileExistsError("输出文件夹已存在，禁止覆盖")

    # 检查图片是否都有标签
    for image in images:
        label = raw_dir / "labels" / f"{image.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(f"缺少标签：{image.name}")

    # 随机划分数据集：80% / 10% / 10%
    random.Random(42).shuffle(images)

    n = len(images)
    a = int(n * 0.8)
    b = round(n * 0.1)

    groups = {
        "train": images[:a],
        "val": images[a:a + b],
        "test": images[a + b:]
    }

    # 复制图片和对应的标签
    for split, files in groups.items():
        images_out = output_dir / "images" / split
        labels_out = output_dir / "labels" / split

        images_out.mkdir(parents=True)
        labels_out.mkdir(parents=True)

        for image in files:
            shutil.copy2(image, images_out / image.name)

            label = raw_dir / "labels" / f"{image.stem}.txt"
            shutil.copy2(label, labels_out / label.name)

        print(f"{split}: {len(files)} 张图片")

    # 生成 data.yaml
    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: red_ball\n"
        "  1: blue_ball\n"
    )

    (output_dir / "data.yaml").write_text(
        yaml_text, encoding="utf-8"
    )

    print("数据预处理完成")


# 2. 数据检查

for split in ("train", "val", "test"):
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    images = {p.stem for p in images_dir.glob("*.jpg")}
    labels = {p.stem for p in labels_dir.glob("*.txt")}

    print(f"{split}: 图片 {len(images)} 张，标签 {len(labels)} 个")

    if not images or images != labels:
        raise ValueError(f"{split} 数据集检查未通过")

print("数据集检查完成")


# 3. 模型训练
#这里只定义了没有写运行代码 因为模型已经训练完了
def train_model():
    model = YOLO("yolo11n.pt")

    model.train(
        data=r"D:\工具\YOLO\data\dataset\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=r"D:\工具\YOLO\runs",
        name="ball_train"
    )


# 4. 模型评估

model = YOLO(r"D:\工具\YOLO\runs\ball_train\weights\best.pt")
print("模型类别：", model.names)

metrics = model.val(
    data=r"D:\工具\YOLO\data\dataset\data.yaml",
    split="test",
    imgsz=640,
    device=0
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)


# 5. 模型推理

results = model.predict(
    source=r"D:\工具\YOLO\data\dataset\images\test",
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
    project=r"D:\工具\YOLO\runs",
    name="final_predict"
)

print(f"推理完成，共处理 {len(results)} 张图片")
