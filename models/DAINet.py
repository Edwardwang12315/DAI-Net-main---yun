# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import torch
import torch.fft
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable , Function
from torchvision import transforms

from layers import *
from data.config import cfg
import cv2
import inspect

import numpy as np
import matplotlib.pyplot as plt


class Interpolate( nn.Module ) :
	# 插值的方法对张量进行上采样或下采样
	def __init__( self , scale_factor ) :
		super( Interpolate , self ).__init__()
		self.scale_factor = scale_factor
	
	def forward( self , x ) :
		x = nn.functional.interpolate( x , scale_factor = self.scale_factor , mode = 'nearest' )
		return x


class FEM( nn.Module ) :
	"""docstring for FEM"""
	
	def __init__( self , in_planes ) :
		super( FEM , self ).__init__()
		inter_planes = in_planes // 3
		inter_planes1 = in_planes - 2 * inter_planes
		self.branch1 = nn.Conv2d( in_planes , inter_planes , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 )
		
		self.branch2 = nn.Sequential(
				nn.Conv2d( in_planes , inter_planes , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) ,
				nn.ReLU( inplace = True ) ,
				nn.Conv2d( inter_planes , inter_planes , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) )
		self.branch3 = nn.Sequential(
				nn.Conv2d( in_planes , inter_planes1 , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) ,
				nn.ReLU( inplace = True ) ,
				nn.Conv2d( inter_planes1 , inter_planes1 , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) ,
				nn.ReLU( inplace = True ) ,
				nn.Conv2d( inter_planes1 , inter_planes1 , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) )
	
	def forward( self , x ) :
		x1 = self.branch1( x )
		x2 = self.branch2( x )
		x3 = self.branch3( x )
		out = torch.cat( (x1 , x2 , x3) , dim = 1 )
		out = F.relu( out , inplace = True )
		return out


# 仿射变换
class SFT_layer( nn.Module ) :
	def __init__( self , in_ch = 3 , inter_ch = 32 , out_ch = 3 , kernel_size = 3 ) :
		super().__init__()
		self.encoder = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) ,
		                              nn.LeakyReLU( True ) , )
		self.decoder = nn.Sequential( nn.Conv2d( inter_ch , out_ch , kernel_size , padding = kernel_size // 2 ) )
		self.shift_conv = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) )
		self.scale_conv = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) )
	
	def forward( self , x , guide ) :
		x = self.encoder( x )
		scale = self.scale_conv( guide )
		shift = self.shift_conv( guide )
		x = x + x * scale + shift
		x = self.decoder( x )
		return x
	
# DEM
class Trans_high( nn.Module ) :
	'''
	通过LAB空间对L进行gamma动态矫正提升亮度与对比度
	利用线性变换降低期望，提高方差
	'''
	
	if True:
		def __init__( self , num_residual_blocks , num_high = 3 ) :
			super( Trans_high , self ).__init__()
			
			self.num_high = num_high
			
			model = [ nn.Conv2d( 9 , 64 , 3 , padding = 1 ) ,
			          nn.LeakyReLU() ]
			
			for _ in range( num_residual_blocks ) :
				model += [ ResidualBlock( 64 ) ]
			
			model += [ nn.Conv2d( 64 , 3 , 3 , padding = 1 ) ]
			
			self.model = nn.Sequential( *model )
			
			for i in range( self.num_high ) :
				trans_mask_block = nn.Sequential(
						nn.Conv2d( 3 , 16 , 1 ) ,
						nn.LeakyReLU() ,
						nn.Conv2d( 16 , 3 , 1 ) )
				setattr( self , 'trans_mask_block_{}'.format( str( i ) ) , trans_mask_block )
		
	if False :
		def __init__( self , ch_blocks = 64 ) :
			super().__init__()
			
			self.contrast = CBAM( ch_blocks )
			self.encoder = nn.Sequential( nn.Conv2d( 3 , 16 , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
			                              nn.Conv2d( 16 , ch_blocks , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
			                              )
			self.decoder = nn.Sequential( nn.Conv2d( ch_blocks , 16 , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
			                              nn.Conv2d( 16 , 3 , 3 , padding = 1 ) , nn.Sigmoid( ) ,
			                              )
	
	if True:
		# x:向上guide pyr:原始金字塔 fake_low:转换后低频
		def forward( self , x , pyr_original , fake_low ) :
			
			pyr_result = [ ]
			pyr_result.append( fake_low )
			mask = self.model( x )
			
			for i in range( self.num_high ) :
				mask = nn.functional.interpolate( mask , size = (
				pyr_original[ -2 - i ].shape[ 2 ] , pyr_original[ -2 - i ].shape[ 3 ]) )
				result_highfreq = torch.mul( pyr_original[ -2 - i ] , mask ) + pyr_original[ -2 - i ]
				self.trans_mask_block = getattr( self , 'trans_mask_block_{}'.format( str( i ) ) )
				result_highfreq = self.trans_mask_block( result_highfreq )
				pyr_result.append( result_highfreq )
				
			return pyr_result
	if False :
		def forward( self , inputs ) :
			x = self.encoder( inputs )
			x = self.contrast( x )
			x= self.decoder( x )
			# 动态计算每个样本的min-max
			min_val = x.view( x.size( 0 ) , -1 ).min( dim = 1 )[ 0 ].view( -1 , 1 , 1 , 1 )
			max_val = x.view( x.size( 0 ) , -1 ).max( dim = 1 )[ 0 ].view( -1 , 1 , 1 , 1 )
			return (x - min_val) / (max_val - min_val + 1e-8)  # 防止除零


# 上采样
class Up_guide( nn.Module ) :
	def __init__( self , kernel_size = 1 , ch = 3 ) :
		super().__init__()
		self.up = nn.Sequential( nn.Upsample( scale_factor = 2 , mode = "bilinear" , align_corners = True ) ,
		                         # 这里的卷积文章中并没有提到，AI认为可以确保引导信息与高频分量的特征在空间和语义上对齐，我持怀疑态度
		                         nn.Conv2d( ch , ch , kernel_size , stride = 1 , padding = kernel_size // 2 ,
		                                    bias = False ) )
	
	def forward( self , x ) :
		return self.up( x )


# 拉普拉斯金字塔
class Lap_Pyramid_Conv( nn.Module ) :
	def __init__( self , num_high = 3 , kernel_size = 5 , channels = 3 ) :
		super().__init__()
		
		self.num_high = num_high
		self.kernel = self.gauss_kernel( kernel_size , channels )
	
	def gauss_kernel( self , kernel_size , channels ) :
		kernel = cv2.getGaussianKernel( kernel_size , 0 ).dot( cv2.getGaussianKernel( kernel_size , 0 ).T )
		kernel = torch.FloatTensor( kernel ).unsqueeze( 0 ).repeat( channels , 1 , 1 , 1 )
		kernel = torch.nn.Parameter( data = kernel , requires_grad = False )
		return kernel
	
	def conv_gauss( self , x , kernel ) :
		n_channels , _ , kw , kh = kernel.shape
		x = torch.nn.functional.pad( x , (kw // 2 , kh // 2 , kw // 2 , kh // 2) ,
		                             mode = 'reflect' )  # replicate    # reflect
		x = torch.nn.functional.conv2d( x , kernel , groups = n_channels )
		return x
	
	def downsample( self , x ) :
		return x[ : , : , : :2 , : :2 ]
	
	def pyramid_down( self , x ) :
		return self.downsample( self.conv_gauss( x , self.kernel ) )
	
	def upsample( self , x ) :
		up = torch.zeros( (x.size( 0 ) , x.size( 1 ) , x.size( 2 ) * 2 , x.size( 3 ) * 2) , device = x.device )
		up[ : , : , : :2 , : :2 ] = x * 4
		
		return self.conv_gauss( up , self.kernel )
	
	def pyramid_decom( self , img ) :
		self.kernel = self.kernel.to( img.device )
		current = img
		pyr = [ ]
		for _ in range( self.num_high ) :
			down = self.pyramid_down( current )
			up = self.upsample( down )
			diff = current - up
			pyr.append( diff )
			current = down
		
		pyr.append( current )
		return pyr
	
	def pyramid_recons( self , pyr ) :
		image = pyr[ 0 ]
		for level in pyr[ 1 : ] :
			up = self.upsample( image )
			image = up + level
		return image

# CGM
class Trans_guide( nn.Module ) :
	# 这里和论文中所说的通道为32有出入
	def __init__( self , ch = 16 ) :
		super().__init__()
		
		self.layer = nn.Sequential( nn.Conv2d( 6 , ch , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
		                            SpatialAttention( 3 ) , nn.Conv2d( ch , 3 , 3 , padding = 1 ) , )
	
	def forward( self , x ) :
		return self.layer( x )


class ChannelAttention( nn.Module ) :
	def __init__( self , in_channels , reduction_ratio = 16 ) :
		super( ChannelAttention , self ).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d( 1 )
		self.max_pool = nn.AdaptiveMaxPool2d( 1 )
		
		# 共享的全连接层
		self.fc = nn.Sequential(
				nn.Linear( in_channels , in_channels // reduction_ratio ) ,
				nn.ReLU() ,
				nn.Linear( in_channels // reduction_ratio , in_channels )
		)
		self.sigmoid = nn.Sigmoid()
	
	def forward( self , x ) :
		# 输入形状: (B, C, H, W)
		avg_out = self.fc( self.avg_pool( x ).squeeze( -1 ).squeeze( -1 ) )  # (B, C)
		max_out = self.fc( self.max_pool( x ).squeeze( -1 ).squeeze( -1 ) )  # (B, C)
		
		# 逐元素相加后激活
		channel_weights = self.sigmoid( avg_out + max_out ).unsqueeze( -1 ).unsqueeze( -1 )  # (B, C, 1, 1)
		return x * channel_weights  # 广播乘法

class SpatialAttention( nn.Module ) :
	def __init__( self , kernel_size = 7 ) :
		super( SpatialAttention , self ).__init__()
		self.conv = nn.Conv2d( 2 , 1 , kernel_size , padding = kernel_size // 2 )
		self.sigmoid = nn.Sigmoid()
	
	def forward( self , x ) :
		# 通道维度的平均和最大池化
		avg_out = torch.mean( x , dim = 1 , keepdim = True )  # (B, 1, H, W)
		max_out , _ = torch.max( x , dim = 1 , keepdim = True )  # (B, 1, H, W)
		
		# 通道拼接
		combined = torch.cat( [ avg_out , max_out ] , dim = 1 )  # (B, 2, H, W)
		spatial_weights = self.sigmoid( self.conv( combined ) )  # (B, 1, H, W)
		return x * spatial_weights  # 广播乘法
	
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)  # 先应用通道注意力
        # print(f'通道注意力值：{x}')
        x = self.spatial_att(x)   # 再应用空间注意力
        # print(f'空间注意力值：{x}')
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

# L通道
class Trans_low( nn.Module ) :
	'''
	通过LAB空间对L进行gamma动态矫正提升亮度与对比度
	利用线性变换降低期望，提高方差
	'''
	def __init__( self , num_residual_blocks ) :
		super().__init__()
		
		if True:
			model = [ nn.Conv2d( 3 , 16 , 3 , padding = 1 ) ,
			          nn.InstanceNorm2d( 16 ) ,
			          nn.LeakyReLU() ,
			          nn.Conv2d( 16 , 64 , 3 , padding = 1 ) ,
			          nn.LeakyReLU() ]
			
			for _ in range( num_residual_blocks ) :
				model += [ ResidualBlock( 64 ) ]
			
			model += [ nn.Conv2d( 64 , 16 , 3 , padding = 1 ) ,
			           nn.LeakyReLU() ,
			           nn.Conv2d( 16 , 3 , 3 , padding = 1 ) ]
			
			self.model = nn.Sequential( *model )
			
		
		if False:
			self.contrast=CBAM(ch_blocks)
			self.encoder = nn.Sequential( nn.Conv2d( 3 , 16 , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
			                              nn.Conv2d( 16 , ch_blocks , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
			                              )
			self.decoder = nn.Sequential(
					nn.Conv2d( ch_blocks , 16 , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
					nn.Conv2d( 16 , 3 , 3 , padding = 1 ) ,nn.LeakyReLU( True ) ,
			)
		
		if False:
			self.gamma_net = nn.Sequential(
					nn.AdaptiveAvgPool2d( 1 ) ,
					nn.Linear( 32 , 32 ) ,  # 输入通道数需匹配
					nn.LeakyReLU() ,
					nn.Linear( 32 , 1 ) ,
					nn.Sigmoid()  # 输出γ∈(0,1)
			)
			self.contrast_net = nn.Sequential(
					nn.Conv2d( 32 , 32 , 3 , padding = 1 ) ,
					nn.InstanceNorm2d( 32 ) ,
					ContextAttention( 32 )  # 上下文注意力模块
			)
			
	def forward( self , inputs ) :
		
		if True:
			out = inputs + self.model( inputs )
			out = torch.tanh( out )
			return out
		
		if False:
			# print(f'LF数据：{inputs}')
			x = self.encoder( inputs )
			# print(f'LF数据编码：{x}')
			x=self.contrast(x)+x # 输出0-2
			# print(f'LF数据加成：{x}')
			x=self.decoder(x)
			# print(f'LF数据解码：{x}')
			# 压缩到0-1 提高了对比度又不会很极端
			min_val = x.view( x.size( 0 ) , -1 ).min( dim = 1 )[ 0 ].view( -1 , 1 , 1 , 1 )
			max_val = x.view( x.size( 0 ) , -1 ).max( dim = 1 )[ 0 ].view( -1 , 1 , 1 , 1 )
			return (x - min_val) / (max_val - min_val + 1e-8)  # 防止除零
		
		if False:
			gamma = self.gamma_net( x ) * 2 + 0.5  # γ∈[0.5, 2.5]
			x = x ** gamma
			x = self.contrast_net( x )
			return x

class DENet( nn.Module ) :
	if True:
		def __init__( self , nrb_low = 5 , nrb_high = 3 , num_high = 3 ) :
			super( ).__init__()
			
			self.lap_pyramid = Lap_Pyramid_Conv( num_high )
			trans_low = Trans_low( nrb_low )
			trans_high = Trans_high( nrb_high , num_high = num_high )
			self.trans_low = trans_low.cuda()
			self.trans_high = trans_high.cuda()
	
	if False:
		def __init__( self , num_high = 3 , ch_blocks = 32 , up_ksize = 1 , high_ch = 32 , high_ksize = 3 , ch_mask = 32 ,
		              gauss_kernel = 5 ) :
			super().__init__()
			self.num_high = num_high
			self.lap_pyramid = Lap_Pyramid_Conv( num_high , gauss_kernel )
			self.trans_low = Trans_low( ch_blocks , ch_mask )
			self.KL = DistillKL( T = 4.0 )
			
			# 这里的setattr和后面的getattr是联动的
			for i in range( 0 , self.num_high ) :
				# self.__setattr__( 'up_guide_layer_{}'.format( i ) , Up_guide( up_ksize , ch = 3 ) )
				self.__setattr__( 'trans_high_layer_{}'.format( i ) , Trans_high( ) )
			# for i in range( 1 , self.num_high ) :
			# 	self.__setattr__( 'down_finetune_layer_{}'.format( i-1 ) , down_finetune( kernel_size = 3 , channels = 3 ) )
		
		# (HF_d + LF_l) VS (HF_l + LF_l) 增强模块提取高频信号效果好；输出图像是亮度良好的。
		
	if True:
		def forward( self , x_dark,x_light ) :
			# HF1 HF2 HF3 LF
			pyrs_d = self.lap_pyramid.pyramid_decom( img = x_dark )
			pyrs_l = self.lap_pyramid.pyramid_decom( img = x_light )
			trans_LF = self.trans_low( pyrs_d[ -1 ] )
			LF_guide = nn.functional.interpolate( pyrs_d[ -1 ] ,
			                                       size = (pyrs_d[ -2 ].shape[ 2 ] , pyrs_d[ -2 ].shape[ 3 ]) )
			LF_B_guide = nn.functional.interpolate( trans_LF ,
			                                       size = (pyrs_d[ -2 ].shape[ 2 ] , pyrs_d[ -2 ].shape[ 3 ]) )
			high_with_low = torch.cat( [ pyrs_d[ -2 ] , LF_guide , LF_B_guide ] , 1 )
			
			pyrs_trans = self.trans_high( high_with_low , pyrs_d , trans_LF )
			
			reconLL_pyrs=[]
			for i in range( len( pyrs_l ) ) :
				reconLL_pyrs.append( pyrs_l[ len( pyrs_l ) - 1 - i ] )  # 取 HF_l + LF_l

			reconDL_pyrs=[]
			reconDL_pyrs.append( pyrs_l[ -1 ] )
			for i in range( len( pyrs_d )-1 ) :
				reconDL_pyrs.append( pyrs_trans[ i+1 ] )  # 取 HF_d + LF_l
			
			reconLD_pyrs=reconLL_pyrs.copy()
			reconLD_pyrs[0] = pyrs_trans[0] # 取 HF_l + LF_d
			
			HF_d_LF_d = self.lap_pyramid.pyramid_recons( pyrs_trans )
			HF_l_LF_l = self.lap_pyramid.pyramid_recons( reconLL_pyrs )
			HF_d_LF_l = self.lap_pyramid.pyramid_recons( reconDL_pyrs )
			HF_l_LF_d = self.lap_pyramid.pyramid_recons( reconLD_pyrs )
			
			return HF_d_LF_d , HF_d_LF_l , HF_l_LF_l ,HF_l_LF_d
		
	if False:
		def forward( self , x_dark , x_light ) :
			# HF1 HF2 HF3 LF
			pyrs_d = self.lap_pyramid.pyramid_decom( img = x_dark )
			pyrs_l = self.lap_pyramid.pyramid_decom( img = x_light )
			
			# image = np.transpose( pyrs_d[ -1 ][ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
			# image = (image * 255).astype( np.uint8 )
			# plt.imshow( image )
			# plt.axis( 'off' )
			# # 保存图像到文件
			# plt.savefig( f'LF_d.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
			
			# for i in range( len(pyrs_d) ) :
			# 	image = np.transpose( pyrs_d[i][ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
			# 	image = (image * 255).astype( np.uint8 )
			# 	plt.imshow( image )
			# 	plt.axis( 'off' )
			# 	# 保存图像到文件
			# 	plt.savefig( f'HF_d_{ i}.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
			
			trans_pyrs = [ ]  # 被处理的金字塔（主通路暗图）
			reconLL_pyrs = [ ]  # 重组金字塔 HF_l LF_l
			reconDL_pyrs = [ ]  # 重组金字塔 HF_d LF_l（原始LF）
			reconLLe_pyrs = [ ]  # 重组金字塔 HF_l LF_l(亮度增强)
			reconLD_pyrs = []
			
			# trans_LF = self.illum_global( img = x_dark , LF = pyrs_d[ -1 ] , level = self.num_high )
			# debug_LF = self.illum_global( img = x_light , LF = pyrs_l[ -1 ] , level = self.num_high )
			
			trans_LF = self.trans_low( pyrs_d[-1] )
			trans_pyrs.append( trans_LF )
			
			# print( f'归一化后的LF:{trans_LF}')
			# image = np.transpose( trans_LF[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
			# image = (image * 255).astype( np.uint8 )
			# plt.imshow( image )
			# plt.axis( 'off' )
			# # 保存图像到文件
			# plt.savefig( f'LF_enh.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
			
			# commom_guide = [ ]
			# for i in range( self.num_high ) :
			# 	guide = self.__getattr__( 'up_guide_layer_{}'.format( i ) )( guide )
			# 	commom_guide.append( guide )
				
				# image = np.transpose( guide[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
				# image = (image * 255).astype( np.uint8 )
				# plt.imshow( image )
				# plt.axis( 'off' )
				# # 保存图像到文件
				# plt.savefig( f'common_guide_{i}.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
			
			for i in range( self.num_high ) :
				trans_HF = self.__getattr__( 'trans_high_layer_{}'.format( i ) )( pyrs_d[ -2 - i ])# , commom_guide[ i ] )
				trans_pyrs.append( trans_HF )  # LF HF3 HF2 HF1
			
				# image = np.transpose( trans_HF[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
				# image = (image * 255).astype( np.uint8 )
				# plt.imshow( image )
				# plt.axis( 'off' )
				# # 保存图像到文件
				# plt.savefig( f'HF_enh_{i}.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
			
			# for i in range( 1 , self.num_high + 1 ) :
			# 	fined_LF = self.__getattr__( 'down_finetune_layer_{}'.format( i - 1 ) )( HF = trans_pyrs[ i ] , level = i ,
			# 	                                                                         LF = trans_pyrs[ 0 ] )
			# 	trans_pyrs[ 0 ] = fined_LF
			
			# for i in range( 1 , self.num_high ) :
			# 	fined_LF = self.__getattr__( 'down_finetune_layer_{}'.format( i - 1 ) )( HF = trans_pyrs[ i ] ,
			# 	                                                                         level = i ,
			# 	                                                                         LF = trans_pyrs[ 0 ] )
			# 	trans_pyrs[ 0 ] = fined_LF
			
	
			# HF_l + LF_l(ori)
			for i in range( len( trans_pyrs ) ) :
				reconLL_pyrs.append( pyrs_l[ len( trans_pyrs ) - 1 - i ] )  # 取 HF_l + LF_l
			
			# HF_l + LF_l(enh)
			# reconLLe_pyrs = reconLL_pyrs.copy()
			# reconLLe_pyrs[ 0 ] = debug_LF
			
			# HF_d + LF_l
			reconDL_pyrs = trans_pyrs.copy()
			reconDL_pyrs[ 0 ] = reconLL_pyrs[ 0 ]
			
			# HF_l + LF_d
			reconLD_pyrs = reconLL_pyrs.copy()
			reconLD_pyrs[ 0 ] = pyrs_d[ -1 ]
			
			# HF_d + LF_d = trans_pyrs
			
			HF_d_LF_d = self.lap_pyramid.pyramid_recons( trans_pyrs )
			HF_l_LF_l = self.lap_pyramid.pyramid_recons( reconLL_pyrs )
			HF_d_LF_l = self.lap_pyramid.pyramid_recons( reconDL_pyrs )
			HF_l_LF_d = self.lap_pyramid.pyramid_recons( reconLD_pyrs )
			# HF_l_LF_le = self.lap_pyramid.pyramid_recons( reconLLe_pyrs )
			
			# print( '最终的DD' )
			# image = np.transpose( HF_d_LF_d[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
			# image = (image * 255).astype( np.uint8 )
			# plt.imshow( image )
			# plt.axis( 'off' )# 保存图像到文件
			# plt.savefig( f'DD.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
		
			# exit()
			return HF_d_LF_d , HF_d_LF_l , HF_l_LF_l ,HF_l_LF_d #, HF_l_LF_le


class VGG16( nn.Module ) :
	def __init__( self , phase , base , extras , fem , head1 , head2 , num_classes ) :
		super().__init__()
		self.phase = phase
		self.num_classes = num_classes
		self.vgg = nn.ModuleList( base )  # 3个vgg16
		
		self.L2Normof1 = L2Norm( 256 , 10 )
		self.L2Normof2 = L2Norm( 512 , 8 )
		self.L2Normof3 = L2Norm( 512 , 5 )
		
		self.extras = nn.ModuleList( extras )
		self.fpn_topdown = nn.ModuleList( fem[ 0 ] )
		self.fpn_latlayer = nn.ModuleList( fem[ 1 ] )
		
		self.fpn_fem = nn.ModuleList( fem[ 2 ] )
		
		self.L2Normef1 = L2Norm( 256 , 10 )
		self.L2Normef2 = L2Norm( 512 , 8 )
		self.L2Normef3 = L2Norm( 512 , 5 )
		
		self.loc_pal1 = nn.ModuleList( head1[ 0 ] )  # nn.ModuleList是一种存储子模块的工具
		self.conf_pal1 = nn.ModuleList( head1[ 1 ] )
		
		self.loc_pal2 = nn.ModuleList( head2[ 0 ] )
		self.conf_pal2 = nn.ModuleList( head2[ 1 ] )
		
		self.KL = DistillKL( T = 4.0 )
		self.mseloss=EnhanceLoss()
		
		if self.phase == 'test' :
			self.softmax = nn.Softmax( dim = -1 )
			self.detect = Detect( cfg )
	
	def forward( self , x , f_DD , f_LL , f_DL , f_LD ) :
		size = x.size()[ 2 : ]
		pal1_sources = list()
		loc_pal1 = list()
		conf_pal1 = list()
		loc_pal2 = list()
		conf_pal2 = list()
		
		loss_enhloss = self.mseloss( f_DD , f_LL , f_DL , f_LD )
		
		for k in range( 16 ) :  # vgg13: 14 vgg16: 16
			x = self.vgg[ k ]( x )
		
		# 这里应该获得1,64,640,640的张量
		for i in range( 5 ) :
			f_DD = self.vgg[ i ]( f_DD )
			f_DL = self.vgg[ i ]( f_DL )  # HF_d_LF_l
			f_LL = self.vgg[ i ]( f_LL )  # HF_l_LF_l
			f_LD = self.vgg[ i ]( f_LD )
		# f_LLe = self.vgg[ i ]( f_LLe )
		
		# 这里要控制LF相同，HF不同，使暗图高频增强有效果
		_f_DD = f_DD.flatten( start_dim = 2 ).mean( dim = -1 )
		_f_DL = f_DL.flatten( start_dim = 2 ).mean( dim = -1 )
		_f_LL = f_LL.flatten( start_dim = 2 ).mean( dim = -1 )
		_f_LD = f_LD.flatten( start_dim = 2 ).mean( dim = -1 )
		# _f_LLe = f_LLe.flatten( start_dim = 2 ).mean( dim = -1 )
		
		loss_mutual = cfg.WEIGHT.MC * (self.KL( _f_DL , _f_LL ) + self.KL( _f_LL , _f_DL )  # 高频优化有效
		                               + self.KL( _f_LD , _f_LL ) + self.KL( _f_LL , _f_LD )  # 低频增强有效
		                               # + self.KL( _f_LL , _f_LLe ) + self.KL( _f_LLe , _f_LL )  # 正常亮度不会提高亮度
		                               )
		
		of1 = x  # 16个vgg后的输出
		s = self.L2Normof1( of1 )
		pal1_sources.append( s )
		# apply vgg up to fc7
		for k in range( 16 , 23 ) :  # vgg13: 14,19 vgg16: 16,23
			x = self.vgg[ k ]( x )
		of2 = x
		s = self.L2Normof2( of2 )
		pal1_sources.append( s )
		
		for k in range( 23 , 30 ) :  # vgg13: 19,24 vgg16: 23,30
			x = self.vgg[ k ]( x )
		of3 = x
		s = self.L2Normof3( of3 )
		pal1_sources.append( s )
		
		for k in range( 30 , len( self.vgg ) ) :  # vgg13: 24 vgg16: 30
			x = self.vgg[ k ]( x )
		of4 = x
		pal1_sources.append( of4 )
		
		for k in range( 2 ) :
			x = F.relu( self.extras[ k ]( x ) , inplace = True )
		of5 = x
		pal1_sources.append( of5 )
		for k in range( 2 , 4 ) :
			x = F.relu( self.extras[ k ]( x ) , inplace = True )
		of6 = x
		pal1_sources.append( of6 )
		
		conv7 = F.relu( self.fpn_topdown[ 0 ]( of6 ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 1 ]( conv7 ) , inplace = True )
		conv6 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 0 ]( of5 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 2 ]( conv6 ) , inplace = True )
		convfc7_2 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 1 ]( of4 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 3 ]( convfc7_2 ) , inplace = True )
		conv5 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 2 ]( of3 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 4 ]( conv5 ) , inplace = True )
		conv4 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 3 ]( of2 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 5 ]( conv4 ) , inplace = True )
		conv3 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 4 ]( of1 ) ) , inplace = True )
		
		ef1 = self.fpn_fem[ 0 ]( conv3 )
		ef1 = self.L2Normef1( ef1 )
		ef2 = self.fpn_fem[ 1 ]( conv4 )
		ef2 = self.L2Normef2( ef2 )
		ef3 = self.fpn_fem[ 2 ]( conv5 )
		ef3 = self.L2Normef3( ef3 )
		ef4 = self.fpn_fem[ 3 ]( convfc7_2 )
		ef5 = self.fpn_fem[ 4 ]( conv6 )
		ef6 = self.fpn_fem[ 5 ]( conv7 )
		
		pal2_sources = (ef1 , ef2 , ef3 , ef4 , ef5 , ef6)
		for (x , l , c) in zip( pal1_sources , self.loc_pal1 , self.conf_pal1 ) :
			loc_pal1.append( l( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			conf_pal1.append( c( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
		
		for (x , l , c) in zip( pal2_sources , self.loc_pal2 , self.conf_pal2 ) :
			loc_pal2.append( l( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			conf_pal2.append( c( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
		
		features_maps = [ ]
		for i in range( len( loc_pal1 ) ) :
			feat = [ ]
			feat += [ loc_pal1[ i ].size( 1 ) , loc_pal1[ i ].size( 2 ) ]
			features_maps += [ feat ]
		
		loc_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in loc_pal1 ] , 1 )
		conf_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in conf_pal1 ] , 1 )
		
		loc_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in loc_pal2 ] , 1 )
		conf_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in conf_pal2 ] , 1 )
		
		priorbox = PriorBox( size , features_maps , cfg , pal = 1 )
		self.priors_pal1 = priorbox.forward().requires_grad_( False )
		priorbox = PriorBox( size , features_maps , cfg , pal = 2 )
		self.priors_pal2 = priorbox.forward().requires_grad_( False )
		
		if self.phase == 'test' :
			output = self.detect.forward( loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) , self.softmax(
					conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) ) ,  # conf preds
			                              self.priors_pal2.type( type( x.data ) ) )
		
		else :
			output = (loc_pal1.view( loc_pal1.size( 0 ) , -1 , 4 ) ,
			          conf_pal1.view( conf_pal1.size( 0 ) , -1 , self.num_classes ) , self.priors_pal1 ,
			          loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) ,
			          conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) , self.priors_pal2)
		
		# out,loss_mutual = self.backbone( x1,Light_dark=x1,Dark_light=Dark_light )
		
		return output , loss_mutual,loss_enhloss
	
	def _upsample_prod( self , x , y ) :
		_ , _ , H , W = y.size()
		return F.interpolate( x , size = (H , W) , mode = 'bilinear' ) * y


class DSFD( nn.Module ) :
	def __init__( self , phase , base , extras , fem , head1 , head2 , num_classes ) :
		super().__init__()
		# enhancement
		self.enhancement = DENet()
		# backbone
		self.backbone=VGG16(phase , base , extras , fem , head1 , head2 , num_classes)
	
	def test_forward( self , x_dark , x_light ) :
		with torch.no_grad() :
			HF_d_LF_d , HF_d_LF_l , HF_l_LF_l , HF_l_LF_d = self.enhancement( x_dark = x_dark ,
			                                                                  x_light = x_light )  # 返回值是归一的
			
			x = HF_d_LF_d
			f_DD = HF_d_LF_d
			f_LL = HF_l_LF_l
			f_DL = HF_d_LF_l
			f_LD = HF_l_LF_d
			# f_LLe = HF_l_LF_le
			
			return self.backbone( x , f_DD , f_LL , f_DL , f_LD )
	
	def forward( self , x_dark , x_light ) :
		
		HF_d_LF_d , HF_d_LF_l , HF_l_LF_l , HF_l_LF_d = self.enhancement( x_dark = x_dark ,
		                                                                   x_light = x_light )  # 返回值是归一的

		x = HF_d_LF_d
		f_DD = HF_d_LF_d
		f_LL = HF_l_LF_l
		f_DL = HF_d_LF_l
		f_LD = HF_l_LF_d
		# f_LLe = HF_l_LF_le
		
		return self.backbone( x,f_DD , f_LL , f_DL , f_LD)
	
	def load_weights( self , base_file ) :
		other , ext = os.path.splitext( base_file )
		if ext == '.pkl' or '.pth' :
			print( 'Loading weights into state dict...' )
			mdata = torch.load( base_file , map_location = lambda storage , loc : storage )
			
			epoch = 50
			self.load_state_dict( mdata )
			print( 'Finished!' )
		else :
			print( 'Sorry only .pth and .pkl files supported.' )
		del other , ext , mdata
		torch.cuda.empty_cache()
		return epoch
	
	def xavier( self , param ) :
		init.xavier_uniform_( param )
	
	def weights_init( self , m ) :
		if isinstance( m , nn.Conv2d ) :
			self.xavier( m.weight.data )
			if 'bias' in m.state_dict().keys() :
				m.bias.data.zero_()
		
		if isinstance( m , nn.ConvTranspose2d ) :
			self.xavier( m.weight.data )
			if 'bias' in m.state_dict().keys() :
				m.bias.data.zero_()
		
		if isinstance( m , nn.BatchNorm2d ) :
			m.weight.data[ ... ] = 1
			m.bias.data.zero_()
	
	def _upsample_prod( self , x , y ) :
		_ , _ , H , W = y.size()
		return F.interpolate( x , size = (H , W) , mode = 'bilinear' ) * y


vgg_cfg = [ 64 , 64 , 'M' , 128 , 128 , 'M' , 256 , 256 , 256 , 'C' , 512 , 512 , 512 , 'M' , 512 , 512 , 512 , 'M' ]
# vgg_cfg = [ 64 , 64 , 'M' , 128 , 128 , 'M' , 256 , 256 , 'C' , 512 , 512 , 'M' , 512 , 512 , 'M' ]
extras_cfg = [ 256 , 'S' , 512 , 128 , 'S' , 256 ]

fem_cfg = [ 256 , 512 , 512 , 1024 , 512 , 256 ]


def fem_module( cfg ) :
	topdown_layers = [ ]
	lat_layers = [ ]
	fem_layers = [ ]
	
	topdown_layers += [ nn.Conv2d( cfg[ -1 ] , cfg[ -1 ] , kernel_size = 1 , stride = 1 , padding = 0 ) ]
	for k , v in enumerate( cfg ) :
		fem_layers += [ FEM( v ) ]
		cur_channel = cfg[ len( cfg ) - 1 - k ]
		if len( cfg ) - 1 - k > 0 :
			last_channel = cfg[ len( cfg ) - 2 - k ]
			topdown_layers += [ nn.Conv2d( cur_channel , last_channel , kernel_size = 1 , stride = 1 , padding = 0 ) ]
			lat_layers += [ nn.Conv2d( last_channel , last_channel , kernel_size = 1 , stride = 1 , padding = 0 ) ]
	return (topdown_layers , lat_layers , fem_layers)


def vgg( cfg , i , batch_norm = False ) :
	layers = [ ]
	in_channels = i
	for v in cfg :
		if v == 'M' :
			layers += [ nn.MaxPool2d( kernel_size = 2 , stride = 2 ) ]
		elif v == 'C' :
			layers += [ nn.MaxPool2d( kernel_size = 2 , stride = 2 , ceil_mode = True ) ]
		else :
			conv2d = nn.Conv2d( in_channels , v , kernel_size = 3 , padding = 1 )
			if batch_norm :
				layers += [ conv2d , nn.BatchNorm2d( v ) , nn.ReLU( inplace = True ) ]
			else :
				layers += [ conv2d , nn.ReLU( inplace = True ) ]
			in_channels = v
	conv6 = nn.Conv2d( 512 , 1024 , kernel_size = 3 , padding = 3 , dilation = 3 )
	conv7 = nn.Conv2d( 1024 , 1024 , kernel_size = 1 )
	layers += [ conv6 , nn.ReLU( inplace = True ) , conv7 , nn.ReLU( inplace = True ) ]
	return layers


def add_extras( cfg , i , batch_norm = False ) :
	# Extra layers added to VGG for feature scaling
	layers = [ ]
	in_channels = i
	flag = False
	for k , v in enumerate( cfg ) :
		if in_channels != 'S' :
			if v == 'S' :
				layers += [ nn.Conv2d( in_channels , cfg[ k + 1 ] , kernel_size = (1 , 3)[ flag ] , stride = 2 ,
				                       padding = 1 ) ]
			else :
				layers += [ nn.Conv2d( in_channels , v , kernel_size = (1 , 3)[ flag ] ) ]
			flag = not flag
		in_channels = v
	return layers


def multibox( vgg , extra_layers , num_classes ) :
	loc_layers = [ ]
	conf_layers = [ ]
	vgg_source = [ 14 , 21 , 28 , -2 ]  # vgg13: [ 12 , 17 , 22 , -2 ] vgg16: [14, 21, 28, -2]
	
	for k , v in enumerate( vgg_source ) :
		# print(v)
		loc_layers += [ nn.Conv2d( vgg[ v ].out_channels , 4 , kernel_size = 3 , padding = 1 ) ]
		conf_layers += [ nn.Conv2d( vgg[ v ].out_channels , num_classes , kernel_size = 3 , padding = 1 ) ]
	for k , v in enumerate( extra_layers[ 1 : :2 ] , 2 ) :
		loc_layers += [ nn.Conv2d( v.out_channels , 4 , kernel_size = 3 , padding = 1 ) ]
		conf_layers += [ nn.Conv2d( v.out_channels , num_classes , kernel_size = 3 , padding = 1 ) ]
	return (loc_layers , conf_layers)


def build_net_dark( phase , num_classes = 2 ) :
	base = vgg( vgg_cfg , 3 )
	extras = add_extras( extras_cfg , 1024 )
	head1 = multibox( base , extras , num_classes )
	head2 = multibox( base , extras , num_classes )
	fem = fem_module( fem_cfg )
	return DSFD( phase , base , extras , fem , head1 , head2 , num_classes )


class DistillKL( nn.Module ) :
	"""KL divergence for distillation"""
	
	# 知识蒸馏模块，处理KL散度
	def __init__( self , T ) :
		super().__init__()
		self.T = T
	
	def forward( self , y_s , y_t ) :
		# y_s学生模型的输出，y_t 教师模型的输出
		p_s = F.log_softmax( y_s / self.T , dim = 1 )  # 对数概率分布
		p_t = F.softmax( y_t / self.T , dim = 1 )  # 概率分布
		# 计算KL散度
		# size_average不使用平均损失，而是返回总损失，(self.T ** 2)补偿温度缩放，/ y_s.shape[0]计算平均损失
		loss = F.kl_div( p_s , p_t , size_average = False ) * (self.T ** 2) / y_s.shape[ 0 ]
		return loss
