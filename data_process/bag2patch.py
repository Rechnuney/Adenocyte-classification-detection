# 导入os和Pillow模块
import os
from PIL import Image

# 定义要处理的文件夹路径
src_dir = "/remote-home/share/DATA/RedHouse/AdenocyteBag/negative_bag_768"
dst_dir = "/remote-home/share/DATA/RedHouse/AdenocyteBag/negative_bag_768_cut"

# 定义裁剪后的图片大小，
size = (64, 64)

# 获取文件夹中的所有文件名，并存入列表
files = os.listdir(src_dir)
i = 0

# 遍历文件列表
for file in files:
    i += 1
    print(f"正在处理第{i}个文件", file)

    # 获取文件的完整路径
    file_path = os.path.join(src_dir, file)
    # 判断是否是图片文件，如果不是则跳过
    if not file_path.lower().endswith((".jpg", ".png", ".bmp")):
        continue
    # 打开图片文件
    img = Image.open(file_path)
    # 获取图片的宽度和高度
    width, height = img.size
    # 计算可以裁剪出多少个小图片，向下取整
    rows = height // size[1]
    cols = width // size[0]
    # 生成同名的子文件夹路径，并创建该文件夹，如果已存在则忽略错误
    # 使用rsplit方法以最后一个'.'为分隔符，取第一个元素作为子文件夹名
    sub_dir = os.path.join(dst_dir, file.rsplit(".", 1)[0])
    os.makedirs(sub_dir, exist_ok=True)
    # 遍历每个小图片的位置，按行列顺序裁剪并保存到子文件夹中，命名为row_col.jpg格式
    for row in range(rows):
        for col in range(cols):
            # 计算当前小图片的左上角和右下角坐标（左，上，右，下）
            left = col * size[0]
            top = row * size[1]
            right = left + size[0]
            bottom = top + size[1]
            box = (left, top, right, bottom)
            # 裁剪当前小图片
            sub_img = img.crop(box)

            #如果patch包含 75% 或更多的白色像素，则该patch将被丢弃

            # 获取图片的像素数据
            sub_img_L = sub_img.convert("L")

            # 定义白色像素的阈值，灰度值大于等于该值的像素被认为是白色
            white_threshold = 240

            # 定义白色像素比例的阈值，0到1之间，超过则删除
            ratio = 0.75
            
            # 计算图片总像素数目
            total_pixels = size[0]*size[1]

            # 计算图片中白色像素数目，即灰度值大于等于阈值的像素数目 
            white_pixels = sum(1 for x in range(size[1]) for y in range(size[0]) if sub_img_L.getpixel((x,y)) >= white_threshold)
            
            # 计算白色像素比例 
            white_ratio = white_pixels / total_pixels 

            #print(white_ratio)
            if white_ratio >= 0.75:
                #print(f"Deleted {file}_{row}_{col} because its white pixel ratio is {white_ratio:.2f}")
                continue
              
            sub_img.save(os.path.join(sub_dir, f"{row}_{col}.png"))
    # 如果文件夹为空则删除该文件夹
    files = os.listdir(sub_dir)
    if len(files) == 0:
        #print("Removing empty folder:", sub_dir)
        os.rmdir(sub_dir)
