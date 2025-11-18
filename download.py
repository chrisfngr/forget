import os
import json
import glob
import subprocess
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import shutil

# 数据集名称
dataset_name = "zhhxte/mllm_cl_textvqa"

# 根据数据集名称创建文件夹（将 / 替换为 _）
dataset_folder_name = dataset_name.replace("/", "_")
data_dir = Path("data") / dataset_folder_name

# 清理旧的未按 repo 组织的数据（避免重复）
old_data_dir = Path("data")
old_images_dir = old_data_dir / "images"
old_json_file = old_data_dir / "dataset.json"

if old_images_dir.exists() and old_images_dir.is_dir():
    print(f"🗑️  清理旧数据: {old_images_dir}")
    shutil.rmtree(old_images_dir)
    print(f"   ✓ 已删除旧 images 目录")

if old_json_file.exists():
    print(f"🗑️  清理旧 JSON 文件: {old_json_file}")
    old_json_file.unlink()
    print(f"   ✓ 已删除旧 dataset.json")

data_dir.mkdir(parents=True, exist_ok=True)
images_dir = data_dir / "images"
images_dir.mkdir(exist_ok=True)

# 登录使用 e.g. `huggingface-cli login` 来访问这个数据集
print("正在加载数据集...")
print(f"数据集: {dataset_name}")
print(f"存储路径: {data_dir}")

# 彻底清理旧的缓存数据（解决 'List' 类型不兼容问题）
print("🗑️  清理 HuggingFace 缓存...")
cache_base = Path.home() / ".cache" / "huggingface" / "datasets"
dataset_key = dataset_name.split("/")[-1]  # mllm_cl_clevr
if cache_base.exists():
    # 使用 shell 命令彻底清理
    subprocess.run(
        f"find {cache_base} -type d -name '*{dataset_key}*' -exec rm -rf {{}} + 2>/dev/null || true",
        shell=True,
        capture_output=True
    )
    print("   ✓ 缓存已清理")

# 加载数据集（强制重新下载）
print("正在下载数据集...")
ds = load_dataset(dataset_name, download_mode="force_redownload")

print(f"数据集结构: {ds}")
print(f"数据集分割: {list(ds.keys())}")

# 查看第一个样本的结构
if len(ds) > 0:
    first_split = list(ds.keys())[0]
    print(f"\n查看 {first_split} 分割的第一个样本:")
    first_sample = ds[first_split][0]
    print(f"字段: {list(first_sample.keys())}")
    print(f"样本内容: {first_sample}")

# 按照 HuggingFace 数据集中的 split 来组织数据
# 为每个 split 创建对应的文件夹和 JSON 文件
split_data = {}  # 存储每个 split 的数据

for split_name, dataset in ds.items():
    print(f"\n处理分割: {split_name}, 样本数: {len(dataset)}")
    
    # 为每个 split 创建对应的图像文件夹
    split_images_dir = images_dir / split_name
    split_images_dir.mkdir(exist_ok=True)
    
    split_data[split_name] = []
    
    for idx, sample in enumerate(dataset):
        # 创建每个样本的文件夹（在对应的 split 文件夹下）
        sample_dir = split_images_dir / f"{idx:06d}"
        sample_dir.mkdir(exist_ok=True)
        
        # 提取字段（根据实际数据集结构调整）
        # 假设字段名为 problem, answer, image 等，需要根据实际情况调整
        problem = sample.get("problem", sample.get("instruction", sample.get("question", "")))
        answer = sample.get("answer", sample.get("output", sample.get("response", "")))
        
        # 处理图像
        image_paths = []
        if "image" in sample:
            image = sample["image"]
            if isinstance(image, Image.Image):
                # 转换为 RGB 模式（如果是 RGBA 或其他模式）
                if image.mode in ("RGBA", "LA", "P"):
                    # 创建白色背景
                    rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                    if image.mode == "P":
                        image = image.convert("RGBA")
                    rgb_image.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
                    image = rgb_image
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                
                # 保存图像
                image_filename = f"image_0.jpg"
                image_path = sample_dir / image_filename
                image.save(image_path, "JPEG")
                # 使用完整的绝对路径
                absolute_path = str(image_path.resolve())
                image_paths.append(absolute_path)
        elif "images" in sample:
            # 如果有多个图像
            images = sample["images"]
            if isinstance(images, list):
                for img_idx, img in enumerate(images):
                    if isinstance(img, Image.Image):
                        # 转换为 RGB 模式
                        if img.mode in ("RGBA", "LA", "P"):
                            rgb_image = Image.new("RGB", img.size, (255, 255, 255))
                            if img.mode == "P":
                                img = img.convert("RGBA")
                            rgb_image.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                            img = rgb_image
                        elif img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        image_filename = f"image_{img_idx}.jpg"
                        image_path = sample_dir / image_filename
                        img.save(image_path, "JPEG")
                        # 使用完整的绝对路径
                        absolute_path = str(image_path.resolve())
                        image_paths.append(absolute_path)
        
        # 构建符合 LLaMA-Factory Alpaca 格式的数据
        data_item = {
            "instruction": problem,
            "input": "",  # 不填（选填）
            "output": answer,
            "images": image_paths
        }
        
        split_data[split_name].append(data_item)
        
        if (idx + 1) % 100 == 0:
            print(f"  已处理 {idx + 1}/{len(dataset)} 个样本")

# 按照 split 保存数据（使用 HuggingFace 数据集中的原始 split）
print(f"\n按照 HuggingFace 数据集中的 split 保存数据...")
total_samples = 0
saved_files = []

# 定义 split 名称映射：train -> cl_train, test -> cl_test
split_name_mapping = {
    "train": "cl_train",
    "test": "cl_test"
}

for split_name, data_list in split_data.items():
    total_samples += len(data_list)
    
    # 使用映射后的文件名，如果没有映射则使用原始名称
    file_name = split_name_mapping.get(split_name, split_name)
    split_file = data_dir / f"{file_name}.json"
    print(f"保存 {split_name} split 到 {split_file} ({len(data_list)} 个样本)...")
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    saved_files.append((split_name, split_file, len(data_list)))

print(f"\n完成！")
print(f"- 数据集: {dataset_name}")
print(f"- 总样本数: {total_samples}")
print(f"- 数据集文件夹: {data_dir}")
print(f"- 图像目录: {images_dir}")
print(f"\n保存的文件:")
for split_name, split_file, count in saved_files:
    print(f"  - {split_name}: {split_file} ({count} 个样本)")
