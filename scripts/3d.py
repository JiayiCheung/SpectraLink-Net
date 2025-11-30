#!/usr/bin/env python3
"""
3D Vessel Visualization using PyVista
生成高质量的3D血管渲染图

使用方法:
    python visualize_3d_pyvista.py --pred_dir /path/to/predictions --gt_dir /path/to/gt --output_dir /path/to/output

依赖安装:
    pip install pyvista numpy scikit-image --break-system-packages
"""

import os
import argparse
import numpy as np

# 设置离屏渲染（集群无显示器时需要）
import pyvista as pv

pv.OFF_SCREEN = True

from skimage import measure


def load_volume(path):
	"""加载npy文件"""
	data = np.load(path)
	if data.ndim == 4:
		# [2, D, H, W] 格式
		return data[1]  # 返回label通道
	return data


def create_mesh_from_volume(volume, threshold=0.5, smooth=True):
	"""
	从3D volume创建mesh

	参数:
		volume: 3D numpy array
		threshold: 二值化阈值
		smooth: 是否平滑mesh

	返回:
		pyvista mesh对象
	"""
	# 二值化
	binary = (volume > threshold).astype(np.float32)
	
	# 检查是否有前景
	if binary.sum() == 0:
		print("警告: volume中没有前景体素")
		return None
	
	# 使用Marching Cubes提取表面
	try:
		verts, faces, normals, values = measure.marching_cubes(
			binary,
			level=0.5,
			spacing=(1.0, 1.0, 1.0),
			step_size=1
		)
	except Exception as e:
		print(f"Marching cubes失败: {e}")
		return None
	
	# 转换为PyVista格式
	# faces需要转换为PyVista格式: [n, i1, i2, i3, n, i1, i2, i3, ...]
	faces_pv = np.hstack([[3] + list(f) for f in faces]).astype(np.int64)
	
	mesh = pv.PolyData(verts, faces_pv)
	
	# 平滑处理
	if smooth:
		mesh = mesh.smooth(n_iter=50, relaxation_factor=0.1)
	
	return mesh


def render_single_vessel(mesh, output_path, title="Vessel", color="red",
                         bg_color="black", camera_position=None):
	"""
	渲染单个血管mesh

	参数:
		mesh: PyVista mesh
		output_path: 输出图片路径
		title: 标题
		color: 血管颜色
		bg_color: 背景颜色
		camera_position: 相机位置
	"""
	plotter = pv.Plotter(off_screen=True, window_size=[1200, 1000])
	plotter.set_background(bg_color)
	
	# 添加mesh
	plotter.add_mesh(
		mesh,
		color=color,
		smooth_shading=True,
		specular=0.5,
		specular_power=15,
		ambient=0.3,
		diffuse=0.7
	)
	
	# 添加标题
	plotter.add_text(title, font_size=14, color="white", position="upper_edge")
	
	# 设置相机
	if camera_position:
		plotter.camera_position = camera_position
	else:
		plotter.camera.zoom(1.2)
	
	# 添加光源
	plotter.add_light(pv.Light(position=(1, 1, 1), intensity=0.8))
	
	# 保存
	plotter.screenshot(output_path)
	plotter.close()
	
	print(f"保存: {output_path}")


def render_comparison(pred_mesh, gt_mesh, output_path, case_id, dice_score=None):
	"""
	渲染预测和GT的对比图

	生成2x2的对比视图:
	- 左上: 预测 (前视图)
	- 右上: GT (前视图)
	- 左下: 预测 (侧视图)
	- 右下: GT (侧视图)
	"""
	# 创建2x2子图
	plotter = pv.Plotter(off_screen=True, shape=(2, 2), window_size=[1600, 1400])
	
	# 标题
	title = f"{case_id}"
	if dice_score is not None:
		title += f" | Dice: {dice_score:.4f}"
	
	# 先计算统一的边界框（确保Pred和GT用相同的视角）
	all_bounds = None
	for mesh in [pred_mesh, gt_mesh]:
		if mesh is not None:
			if all_bounds is None:
				all_bounds = list(mesh.bounds)
			else:
				bounds = mesh.bounds
				all_bounds[0] = min(all_bounds[0], bounds[0])  # xmin
				all_bounds[1] = max(all_bounds[1], bounds[1])  # xmax
				all_bounds[2] = min(all_bounds[2], bounds[2])  # ymin
				all_bounds[3] = max(all_bounds[3], bounds[3])  # ymax
				all_bounds[4] = min(all_bounds[4], bounds[4])  # zmin
				all_bounds[5] = max(all_bounds[5], bounds[5])  # zmax
	
	# 计算中心点
	if all_bounds:
		center = [
			(all_bounds[0] + all_bounds[1]) / 2,
			(all_bounds[2] + all_bounds[3]) / 2,
			(all_bounds[4] + all_bounds[5]) / 2
		]
		# 计算合适的相机距离
		max_range = max(
			all_bounds[1] - all_bounds[0],
			all_bounds[3] - all_bounds[2],
			all_bounds[5] - all_bounds[4]
		)
		dist = max_range * 2.5
	else:
		center = [0, 0, 0]
		dist = 300
	
	# 定义统一的相机位置
	# 前视图: 从Y负方向看
	front_cam = [(center[0], center[1] - dist, center[2]), center, (0, 0, 1)]
	# 侧视图: 从X正方向看
	side_cam = [(center[0] + dist, center[1], center[2]), center, (0, 0, 1)]
	# 俯视图: 从Z正方向看
	top_cam = [(center[0], center[1], center[2] + dist), center, (0, 1, 0)]
	# 斜视图: 45度角
	iso_cam = [(center[0] + dist * 0.7, center[1] - dist * 0.7, center[2] + dist * 0.5), center, (0, 0, 1)]
	
	# 左上: 预测 斜视图（最能展示3D结构）
	plotter.subplot(0, 0)
	plotter.set_background("black")
	if pred_mesh is not None:
		plotter.add_mesh(pred_mesh, color="orangered", smooth_shading=True,
		                 specular=0.5, ambient=0.3)
	plotter.camera_position = iso_cam
	plotter.add_text("Prediction (3D View)", font_size=11, color="white")
	
	# 右上: GT 斜视图
	plotter.subplot(0, 1)
	plotter.set_background("black")
	if gt_mesh is not None:
		plotter.add_mesh(gt_mesh, color="gold", smooth_shading=True,
		                 specular=0.5, ambient=0.3)
	plotter.camera_position = iso_cam
	plotter.add_text("Ground Truth (3D View)", font_size=11, color="white")
	
	# 左下: 预测 前视图
	plotter.subplot(1, 0)
	plotter.set_background("black")
	if pred_mesh is not None:
		plotter.add_mesh(pred_mesh, color="orangered", smooth_shading=True,
		                 specular=0.5, ambient=0.3)
	plotter.camera_position = front_cam
	plotter.add_text("Prediction (Front)", font_size=11, color="white")
	
	# 右下: GT 前视图
	plotter.subplot(1, 1)
	plotter.set_background("black")
	if gt_mesh is not None:
		plotter.add_mesh(gt_mesh, color="gold", smooth_shading=True,
		                 specular=0.5, ambient=0.3)
	plotter.camera_position = front_cam
	plotter.add_text("Ground Truth (Front)", font_size=11, color="white")
	
	# 不添加总标题（case名字），保持画面干净
	
	plotter.screenshot(output_path)
	plotter.close()
	
	print(f"保存对比图: {output_path}")


def render_overlay(pred_mesh, gt_mesh, output_path, case_id):
	"""
	渲染叠加视图 - 预测和GT重叠显示

	- 绿色: 重叠部分 (TP)
	- 红色: 仅预测 (FP)
	- 蓝色: 仅GT (FN)
	"""
	plotter = pv.Plotter(off_screen=True, window_size=[1400, 1200])
	plotter.set_background("black")
	
	# 添加预测 (半透明红色)
	if pred_mesh is not None:
		plotter.add_mesh(pred_mesh, color="red", opacity=0.5,
		                 smooth_shading=True, label="Prediction")
	
	# 添加GT (半透明蓝色)
	if gt_mesh is not None:
		plotter.add_mesh(gt_mesh, color="cyan", opacity=0.5,
		                 smooth_shading=True, label="Ground Truth")
	
	# 添加图例
	plotter.add_legend(bcolor="black", face="circle")
	
	# 标题
	plotter.add_text(f"{case_id} - Overlay View", font_size=14, color="white")
	
	# 自动调整相机
	plotter.camera.zoom(1.2)
	
	plotter.screenshot(output_path)
	plotter.close()
	
	print(f"保存叠加图: {output_path}")


def render_rotating_gif(mesh, output_path, title="Vessel", color="red", n_frames=36):
	"""
	生成旋转的GIF动画

	参数:
		mesh: PyVista mesh
		output_path: 输出GIF路径
		title: 标题
		color: 颜色
		n_frames: 帧数
	"""
	plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
	plotter.set_background("black")
	
	plotter.add_mesh(mesh, color=color, smooth_shading=True,
	                 specular=0.5, ambient=0.3)
	plotter.add_text(title, font_size=12, color="white")
	
	# 打开GIF写入
	plotter.open_gif(output_path)
	
	# 旋转一圈
	for i in range(n_frames):
		plotter.camera.azimuth = i * (360 / n_frames)
		plotter.write_frame()
	
	plotter.close()
	print(f"保存GIF: {output_path}")


def process_case(pred_path, gt_path, output_dir, case_id, dice_score=None):
	"""处理单个case"""
	print(f"\n处理: {case_id}")
	
	# 加载数据
	pred_vol = load_volume(pred_path)
	print(f"  预测shape: {pred_vol.shape}, 前景比例: {(pred_vol > 0.5).mean():.4f}")
	
	gt_vol = None
	if gt_path and os.path.exists(gt_path):
		gt_data = np.load(gt_path)
		print(f"  GT原始shape: {gt_data.shape}, dtype: {gt_data.dtype}")
		
		if gt_data.ndim == 4:
			# [2, D, H, W] 格式 - 第0通道是图像，第1通道是标签
			label_channel = gt_data[1]
			unique_labels = np.unique(label_channel)
			print(f"  GT标签唯一值: {unique_labels}")
			
			# 如果标签只有0和1，直接用；如果有0,1,2则只取1（血管）
			if 2 in unique_labels:
				gt_vol = (label_channel == 1).astype(np.float32)
				print(f"  过滤后只保留label=1 (血管)")
			else:
				gt_vol = (label_channel > 0.5).astype(np.float32)
		else:
			gt_vol = (gt_data > 0.5).astype(np.float32)
		
		print(f"  GT shape: {gt_vol.shape}, 前景比例: {(gt_vol > 0.5).mean():.4f}")
		
		if (gt_vol > 0.5).sum() == 0:
			print(f"  警告: GT中没有前景！")
			gt_vol = None
	else:
		print(f"  GT文件不存在或路径为空: {gt_path}")
	
	# 创建mesh
	print("  创建预测mesh...")
	pred_mesh = create_mesh_from_volume(pred_vol, smooth=True)
	
	gt_mesh = None
	if gt_vol is not None:
		print("  创建GT mesh...")
		gt_mesh = create_mesh_from_volume(gt_vol, smooth=True)
	
	# 渲染
	os.makedirs(output_dir, exist_ok=True)
	
	# 1. 单独渲染预测
	if pred_mesh is not None:
		render_single_vessel(
			pred_mesh,
			os.path.join(output_dir, f"{case_id}_pred_3d.png"),
			title=f"{case_id} - Prediction",
			color="orangered"
		)
	
	# 2. 单独渲染GT
	if gt_mesh is not None:
		render_single_vessel(
			gt_mesh,
			os.path.join(output_dir, f"{case_id}_gt_3d.png"),
			title=f"{case_id} - Ground Truth",
			color="gold"
		)
	
	# 3. 对比图
	if pred_mesh is not None or gt_mesh is not None:
		render_comparison(
			pred_mesh, gt_mesh,
			os.path.join(output_dir, f"{case_id}_comparison_3d.png"),
			case_id, dice_score
		)
	
	# 4. 叠加图
	if pred_mesh is not None and gt_mesh is not None:
		render_overlay(
			pred_mesh, gt_mesh,
			os.path.join(output_dir, f"{case_id}_overlay_3d.png"),
			case_id
		)
	
	# 5. 旋转GIF (可选，比较慢)
	# if pred_mesh is not None:
	#     render_rotating_gif(
	#         pred_mesh,
	#         os.path.join(output_dir, f"{case_id}_rotation.gif"),
	#         title=case_id,
	#         color="orangered"
	#     )
	
	print(f"  完成: {case_id}")


def main():
	parser = argparse.ArgumentParser(description='3D Vessel Visualization with PyVista')
	parser.add_argument('--pred_dir', type=str, required=True,
	                    help='预测结果目录 (包含*_pred.npy文件)')
	parser.add_argument('--gt_dir', type=str, default=None,
	                    help='GT数据目录 (可选)')
	parser.add_argument('--output_dir', type=str, required=True,
	                    help='输出目录')
	parser.add_argument('--cases', type=str, nargs='+', default=None,
	                    help='指定要处理的case ID列表')
	parser.add_argument('--max_cases', type=int, default=None,
	                    help='最多处理的case数量')
	
	args = parser.parse_args()
	
	# 获取所有预测文件
	pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('_pred.npy')])
	
	if args.cases:
		pred_files = [f for f in pred_files if any(c in f for c in args.cases)]
	
	if args.max_cases:
		pred_files = pred_files[:args.max_cases]
	
	print(f"找到 {len(pred_files)} 个预测文件")
	
	# 处理每个case
	for pred_file in pred_files:
		case_id = pred_file.replace('_pred.npy', '')
		pred_path = os.path.join(args.pred_dir, pred_file)
		
		# 查找GT
		gt_path = None
		if args.gt_dir:
			gt_path = os.path.join(args.gt_dir, f"{case_id}.npy")
			if not os.path.exists(gt_path):
				gt_path = None
		
		process_case(pred_path, gt_path, args.output_dir, case_id)
	
	print(f"\n全部完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
	main()