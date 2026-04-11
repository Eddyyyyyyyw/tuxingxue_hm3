import taichi as ti
import numpy as np

# 使用 gpu 后端
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000 # 曲线采样点数量

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# --- 【性能优化核心 1】：新增一个用于存放曲线坐标的 GPU 缓冲区 ---
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# 曲线模式：0 为贝塞尔曲线，1 为 B 样条曲线
current_mode = 0

def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

def evaluate_b_spline(points, t):
    """计算均匀三次 B 样条曲线上的点"""
    n = len(points) - 1
    p = 3  # 三次 B 样条
    
    # 生成节点向量
    knots = [0.0] * (p + 1) + [float(i) / (n - p) for i in range(n - p + 1)] + [1.0] * (p + 1)
    
    # 找到当前 t 所在的区间
    span = 0
    for i in range(p, n + 1):
        if t <= knots[i + 1]:
            span = i
            break
    
    # 计算基函数值
    N = [0.0] * (p + 1)
    N[0] = 1.0
    
    for j in range(1, p + 1):
        left = [0.0] * (p + 1)
        right = [0.0] * (p + 1)
        
        for i in range(j, p + 1):
            left[i] = t - knots[i - j]
            right[i] = knots[i + 1] - t
        
        for i in range(j, p + 1):
            saved = 0.0
            for k in range(i - j + 1, i + 1):
                den = knots[k + j] - knots[k]
                if den == 0:
                    continue
                temp = N[k] / den
                N[k] = saved + right[k] * temp
                saved = left[k] * temp
    
    # 计算曲线点
    x, y = 0.0, 0.0
    for i in range(span - p, span + 1):
        basis = N[i - (span - p)]
        x += points[i][0] * basis
        y += points[i][1] * basis
    
    return [x, y]

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# --- 【性能优化核心 2】：将“点亮像素”的工作交给 GPU 并行执行 ---
@ti.kernel
def draw_curve_kernel(n: ti.i32, mode: ti.i32):
    # 这个 for 循环在 kernel 中，Taichi 会自动将其在 GPU 上极速执行
    for i in range(n):
        pt = curve_points_field[i]
        # 计算浮点像素坐标（亚像素精度）
        x_float = pt[0] * WIDTH
        y_float = pt[1] * HEIGHT
        
        # 计算整数像素坐标
        x_int = ti.cast(x_float, ti.i32)
        y_int = ti.cast(y_float, ti.i32)
        
        # 反走样：检查 3x3 像素邻域
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                neighbor_x = x_int + dx
                neighbor_y = y_int + dy
                
                # 检查边界
                if 0 <= neighbor_x < WIDTH and 0 <= neighbor_y < HEIGHT:
                    # 计算像素中心点与曲线点的距离
                    pixel_center_x = neighbor_x + 0.5
                    pixel_center_y = neighbor_y + 0.5
                    distance = ti.sqrt((pixel_center_x - x_float) ** 2 + (pixel_center_y - y_float) ** 2)
                    
                    # 基于距离计算权重（距离越近，权重越大）
                    if distance < 1.5:
                        weight = max(0.0, 1.0 - distance * 0.7)
                        # 根据曲线模式设置颜色
                        if mode == 0:  # 贝塞尔曲线 - 绿色
                            pixels[neighbor_x, neighbor_y] += ti.Vector([0.0, weight, 0.0])
                        else:  # B 样条曲线 - 蓝色
                            pixels[neighbor_x, neighbor_y] += ti.Vector([0.0, 0.0, weight])

def main():
    global current_mode
    window = ti.ui.Window("Bezier Curve (60 FPS Restored)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"Added control point: {pos}")
            elif e.key == 'c':
                control_points = []
                print("Canvas cleared.")
            elif e.key == 'b':
                # 切换曲线模式
                current_mode = 1 - current_mode
                mode_name = "B-spline" if current_mode == 1 else "Bezier"
                window.title = f"{mode_name} Curve (60 FPS Restored)"
                print(f"Switched to {mode_name} mode")
        
        clear_pixels()
        
        current_count = len(control_points)
        if current_count >= 2:
            # 1. 在 CPU 端 (Python) 把所有点的坐标算好，存进 numpy 数组
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                if current_mode == 0:  # 贝塞尔曲线
                    curve_points_np[t_int] = de_casteljau(control_points, t)
                else:  # B 样条曲线
                    if current_count >= 4:
                        curve_points_np[t_int] = evaluate_b_spline(control_points, t)
                    else:
                        # 控制点不足 4 个时，回退到贝塞尔曲线
                        curve_points_np[t_int] = de_casteljau(control_points, t)
            
            # 2. 一次性打包发送给 GPU (只发生 1 次内存通信，而不是 1000 次)
            curve_points_field.from_numpy(curve_points_np)
            
            # 3. 呼叫 GPU：数据已经提供，去显存里将对应像素涂绿
            draw_curve_kernel(NUM_SEGMENTS + 1, current_mode)
                    
        canvas.set_image(pixels)
        
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()

if __name__ == '__main__':
    main()
