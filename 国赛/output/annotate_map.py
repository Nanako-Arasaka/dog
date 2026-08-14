from PIL import Image, ImageDraw, ImageFont
import math

img = Image.open("/Users/silencecf/code/DOG/dog_repo/国赛/output/pdf_extract/map_hi.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 28)
small = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 22)

# 标注点：按 FSM 顺序
points = [
    ("1 start_exit", 890, 2080, "#2ecc71"),
    ("2 obstacle_entry", 890, 1980, "#e67e22"),
    ("3 obstacle_exit", 890, 1550, "#e67e22"),
    ("4 box1_side1", 780, 740, "#3498db"),
    ("5 box1_side2", 510, 990, "#3498db"),
    ("6 box2_side1", 780, 1220, "#3498db"),
    ("7 box2_side2", 510, 1030, "#3498db"),
    ("8 pick_area", 1150, 1230, "#9b59b6"),
    ("9 place_A", 880, 400, "#e74c3c"),
    ("10 place_B", 1050, 400, "#e74c3c"),
    ("11 place_C", 1210, 400, "#e74c3c"),
    ("12 place_D", 1380, 400, "#e74c3c"),
    ("13 finish", 890, 320, "#2ecc71"),
]

r = 18
for label, x, y, color in points:
    # 外圈
    draw.ellipse([x-r-3, y-r-3, x+r+3, y+r+3], outline="#2c3e50", width=3)
    # 内圆
    draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    # 标签：右侧偏移，避免重叠
    tx = x + 35
    ty = y - 14
    # 黑色描边提高可读性
    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        draw.text((tx+dx, ty+dy), label, font=font, fill="#000000")
    draw.text((tx, ty), label, font=font, fill="#ffffff")

# 图例
legend_x, legend_y = 100, 2150
legend_items = [
    ("起点/终点", "#2ecc71"),
    ("障碍区", "#e67e22"),
    ("巡检箱", "#3498db"),
    ("抓取/投放", "#9b59b6"),
    ("投放区 A-D", "#e74c3c"),
]
draw.rectangle([legend_x-10, legend_y-10, legend_x+260, legend_y+180], fill="#ffffff", outline="#2c3e50", width=2)
for i, (text, color) in enumerate(legend_items):
    ly = legend_y + i * 34
    draw.ellipse([legend_x, ly, legend_x+20, ly+20], fill=color, outline="#2c3e50", width=2)
    draw.text((legend_x+30, ly-2), text, font=small, fill="#000000")

img.save("/Users/silencecf/code/DOG/dog_repo/国赛/output/waypoints_on_map.png")
print("saved waypoints_on_map.png")
