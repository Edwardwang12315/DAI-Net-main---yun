# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import torch
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
	def __init__( self , weight_init , in_ch = 3 , inter_ch = 32 , out_ch = 3 , kernel_size = 3 ) :
		super().__init__()
		self.encoder = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) ,
		                              nn.Sigmoid(  ) , )
		self.decoder = nn.Sequential( nn.Conv2d( inter_ch , out_ch , kernel_size , padding = kernel_size // 2 ) )
		self.shift_conv = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) )
		self.scale_conv = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) )

		# self.encoder.apply( weight_init )
		# self.decoder.apply( weight_init )
		# self.shift_conv.apply( weight_init )
		# self.scale_conv.apply( weight_init )

	def forward( self , x , guide ) :
		x = self.encoder( x )
		scale = self.scale_conv( guide )
		shift = self.shift_conv( guide )
		# with torch.no_grad() :
		x =x+ (x * scale + shift)
		x = self.decoder( x )

		return x


# DEM
class Trans_high( nn.Module ) :
	def __init__( self , weight_init , in_ch = 3 , inter_ch = 32 , out_ch = 3 , kernel_size = 3 ) :
		super().__init__()

		self.sft = SFT_layer( weight_init = weight_init , in_ch = in_ch , inter_ch = inter_ch , out_ch = out_ch ,
		                      kernel_size = kernel_size )

	def forward( self , x , guide ) :
		out = x + self.sft( x , guide )

		return out


# 上采样
class Up_guide( nn.Module ) :
	def __init__( self , weight_init , kernel_size = 1 , ch = 3 ) :
		super().__init__()
		self.up = nn.Sequential( nn.Upsample( scale_factor = 2 , mode = "bilinear" , align_corners = True ) ,
		                         # 这里的卷积文章中并没有提到，AI认为可以确保引导信息与高频分量的特征在空间和语义上对齐，我持怀疑态度
		                         nn.Conv2d( ch , ch , kernel_size , stride = 1 , padding = kernel_size // 2 ) )

		# self.up.apply( weight_init )

	def forward( self , x ) :

		return self.up( x )


# 拉普拉斯金字塔
class Lap_Pyramid_Conv() :
	def __init__( self , num_high = 3 , kernel_size = 5 , channels = 3 ) :
		super().__init__()

		self.num_high = num_high
		self.kernel = self.gauss_kernel( kernel_size , channels )
		self.transform=transforms.ToPILImage()

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
		out = self.downsample( self.conv_gauss( x , self.kernel ) )

		return out

	def upsample( self , x ) :
		up = torch.zeros( (x.size( 0 ) , x.size( 1 ) , x.size( 2 ) * 2 , x.size( 3 ) * 2) , device = x.device )
		up[ : , : , : :2 , : :2 ] = (x * 4)
		out = self.conv_gauss( up , self.kernel )

		return out

	def pyramid_decom( self , img ) :
		self.kernel = self.kernel.to( img.device )
		current = img
		pyr = [ ]
		for i in range( self.num_high ) :
			down = self.pyramid_down( current )
			up = self.upsample( down )
			diff = current - up
			pyr.append( diff )
			current = down

			# cut = self.transform( diff.squeeze( 0 ) )
			# cut.save( f'level_{i}.png' )

		pyr.append( current )  # HF1 HF2 HF3 HF4 LF

		# cut = self.transform( current.squeeze(0) )
		# cut.save( f'level_{self.num_high}.png' )

		return pyr

	def pyramid_recons( self , pyr ) :
		# with torch.no_grad() :
		image = pyr[ 0 ]
		# for level in pyr[ 1 : ] :
		for i in range( len( pyr ) - 1 ) :
			level = pyr[ 1+i ]
			up = self.upsample( image )
			image = up + level

			# recon = self.transform( image.squeeze( 0 ) )
			# recon.save( f'recon_level_{len( pyr ) - 1 - i}.png' )
		# exit()
		
		return image


# CGM
class Trans_guide( nn.Module ) :
	# 这里和论文中所说的通道为32有出入
	def __init__( self , weight_init , ch = 16 ) :
		super().__init__()
		self.layer = nn.Sequential( nn.Conv2d( 6 , ch , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
		                            SpatialAttention_plusx( weight_init = weight_init , kernel_size = 3 ) ,
		                            nn.Conv2d( ch , 3 , 3 , padding = 1 ) , )

		# self.layer.apply( weight_init )

	def forward( self , x ) :
		out = self.layer( x )

		return out


# L通道
class Trans_low( nn.Module ) :
	def __init__( self , weight_init , ch_blocks = 64 , ch_mask = 16 ) :
		super().__init__()

		self.encoder = nn.Sequential( nn.Conv2d( 3 , 16 , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
		                              nn.Conv2d( 16 , ch_blocks , 3 , padding = 1 ) , nn.LeakyReLU( True ) )

		self.mm1 = nn.Sequential( nn.Conv2d( ch_blocks , ch_blocks // 4 , kernel_size = 1 , padding = 0 ) )
		self.mm2 = nn.Sequential( nn.Conv2d( ch_blocks , ch_blocks // 4 , kernel_size = 3 , padding = 3 // 2 ) )
		self.mm3 = nn.Sequential( nn.Conv2d( ch_blocks , ch_blocks // 4 , kernel_size = 5 , padding = 5 // 2 ) )
		self.mm4 = nn.Sequential( nn.Conv2d( ch_blocks , ch_blocks // 4 , kernel_size = 7 , padding = 7 // 2 ) )

		self.decoder = nn.Sequential( nn.Conv2d( ch_blocks , 16 , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
		                              nn.Conv2d( 16 , 3 , 3 , padding = 1 ) )

		self.trans_guide = Trans_guide( weight_init = weight_init , ch = ch_mask )

		# self.encoder.apply( weight_init )
		# self.decoder.apply( weight_init )
		# self.mm1.apply( weight_init )
		# self.mm2.apply( weight_init )
		# self.mm3.apply( weight_init )
		# self.mm4.apply( weight_init )

	def forward( self , x ) :
		x1 = self.encoder( x )  # 这里是不是应该对应mm1-4
		x1_1 = self.mm1( x1 )
		x1_2 = self.mm2( x1 )
		x1_3 = self.mm3( x1 )
		x1_4 = self.mm4( x1 )

		x1 = torch.cat( [ x1_1 , x1_2 , x1_3 , x1_4 ] , dim = 1 )
		x1 = self.decoder( x1 )

		# with torch.no_grad() :
		out = (x + x1)
		out = torch.relu( out )  # 当前通道输出

		mask = self.trans_guide( torch.cat( [ x , out ] , dim = 1 ) )  # 输出给high-layer

		return out , mask


class illumination_global( nn.Module ) :
	def __init__( self , weight_init , kernel_size = 7 ) :
		super().__init__()
		# assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
		padding = 3 if kernel_size == 7 else 1
		self.downsample = nn.Sequential( nn.Conv2d( 3 , 3 , kernel_size , padding = padding , stride = 2 ) ,
		                                 nn.ReLU( True ) , )
		self.encoder = nn.Sequential( nn.Conv2d( 3 , 3 , kernel_size , padding = padding ) , nn.ReLU( True ) ,
		                                   nn.Conv2d(3,32,kernel_size,padding=padding),nn.LeakyReLU(True))

		self.shift_conv = nn.Sequential( nn.Conv2d( 3 , 32 , kernel_size , padding = kernel_size // 2 ) )
		self.scale_conv = nn.Sequential( nn.Conv2d( 3 , 32 , kernel_size , padding = kernel_size // 2 ) )
		self.decoder = nn.Sequential( nn.Conv2d( 32 , 3 , kernel_size , padding = kernel_size // 2 ) )

		# self.downsample.apply( weight_init )
		# self.encoder.apply( weight_init )
		# self.shift_conv.apply( weight_init )
		# self.scale_conv.apply( weight_init )
		# self.decoder.apply( weight_init )

	def forward( self , img , LF,level ) :
		# 下采样
		for i in range( level ) :  # LF HF3 HF2 HF1 0 1 2 3
			img = self.downsample( img )
		# 提取光照特征
		scale=self.encoder(LF)
		scale=self.decoder(scale)
		LF=LF*scale+LF

		return LF

class down_finetune( nn.Module):
	def __init__(self,weight_init,kernel_size=3,channels=3):
		super().__init__()
		self.conv=nn.Sequential(nn.Conv2d(2,1,kernel_size,padding=1),nn.Sigmoid())
		self.downsample=nn.Sequential(nn.Conv2d(channels,channels,kernel_size,padding=1,stride = 2),nn.Sigmoid())

		# self.conv.apply(weight_init)
		# self.downsample.apply(weight_init)

	def forward( self ,HF,level,LF):
		avg_out = torch.mean( HF , dim = 1 , keepdim = True )
		max_out , _ = torch.max( HF , dim = 1 , keepdim = True )
		attention = torch.cat( [ avg_out , max_out ] , dim = 1 )

		HF = (self.conv( attention ) * HF)+HF
		for i in range( level ) :  # LF HF3 HF2 HF1 0 1 2 3
			HF = self.downsample( HF )
		out=LF+HF

		return out

class SpatialAttention_plusx( nn.Module ) :
	def __init__( self , weight_init , kernel_size = 7 ) :
		super().__init__()
		# assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
		padding = 3 if kernel_size == 7 else 1
		self.conv = nn.Sequential( nn.Conv2d( 2 , 1 , kernel_size , padding = padding ) ,nn.Sigmoid() )

		# self.conv.apply( weight_init )

	def forward( self , x ) :
		# 1*h*w
		avg_out = torch.mean( x , dim = 1 , keepdim = True )
		max_out , _ = torch.max( x , dim = 1 , keepdim = True )
		attention = torch.cat( [ avg_out , max_out ] , dim = 1 )
		# 1*h*w
		out = (self.conv( attention ) * x)
		return out


class DSFD( nn.Module ) :
	"""Single Shot Multibox Architecture
	The network is composed of a base VGG network followed by the
	added multibox conv layers.  Each multibox layer branches into
		1) conv2d for class conf scores
		2) conv2d for localization predictions
		3) associated priorbox layer to produce default bounding
		   boxes specific to the layer's feature map size.
	See: https://arxiv.org/pdf/1512.02325.pdf for more details.

	Args:
		phase: (string) Can be "test" or "train"
		size: input image size
		base: VGG16 layers for input, size of either 300 or 500
		extras: extra layers that feed to multibox loc and conf layers
		head: "multibox head" consists of loc and conf conv layers
	"""

	def __init__( self , phase , base , extras , fem , head1 , head2 , num_classes ) :
		super( DSFD , self ).__init__()
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

		self.num_high = 3
		self.lap_pyramid = Lap_Pyramid_Conv( self.num_high , kernel_size = 5 )  # 不需要梯度
		self.pipline_LF = Trans_low( weight_init = self.weights_init , ch_blocks = 32 , ch_mask = 32 )
		self.illumination_global = illumination_global( weight_init = self.weights_init , kernel_size = 3 )
		# 这里的setattr和后面的getattr是联动的
		for i in range( 0 , self.num_high ) :
			# LF向上传递
			self.__setattr__( 'up_guide_layer_{}'.format( i ) ,
			                  Up_guide( weight_init = self.weights_init , kernel_size = 1 , ch = 3 ) )
			# HFs完整通路
			self.__setattr__( 'pipline_HFs{}'.format( i ) ,
			                  Trans_high( weight_init = self.weights_init , in_ch = 3 , inter_ch = 32 , out_ch = 3 ,
			                              kernel_size = 3 , ) )
			# HFs向下 细节增强
			self.__setattr__( 'down_guide_attention{}'.format( i ) ,
			                  down_finetune( weight_init = self.weights_init , kernel_size = 3 , ) )

		self.KL = DistillKL( T = 4.0 )

		if self.phase == 'test' :
			self.softmax = nn.Softmax( dim = -1 )
			self.detect = Detect( cfg )

	def _upsample_prod( self , x , y ) :
		_ , _ , H , W = y.size()
		return F.interpolate( x , size = (H , W) , mode = 'bilinear' ) * y

	# 反射图解码通路
	def enh_forward( self , x ) :

		x = x[ :1 ]
		for k in range( 5 ) :
			x = self.vgg[ k ]( x )

		R = self.ref( x )

		return R

	def test_forward( self , x_dark):# , x_light ) :
		with torch.no_grad() :
			size = x_dark.size()[ 2 : ]
			pal1_sources = list()
			pal2_sources = list()
			loc_pal1 = list()
			conf_pal1 = list()
			loc_pal2 = list()
			conf_pal2 = list()

			pyrs_dark = self.lap_pyramid.pyramid_decom( img = x_dark )  # HF1 HF2 HF3 HF4 LF

			# 主通路(暗图)
			Lap_pyrs_dark = [ ]
			commom_guide = [ ]
			# 跨频率指导
			LF , guide = self.pipline_LF( pyrs_dark[ -1 ] )  # LF通道的两个返回值
			Lap_pyrs_dark.append( LF )  # LF
			# LF向上传递
			for i in range( self.num_high ) :
				guide = self.__getattr__( 'up_guide_layer_{}'.format( i ) )( guide )
				commom_guide.append( guide )
			# HFs完整通路
			for i in range( self.num_high ) :
				# 高频信号通路,从HF4开始
				HFs = self.__getattr__( 'pipline_HFs{}'.format( i ) )( pyrs_dark[ -2 - i ] ,
				                                                       commom_guide[ i ] )  # HF4 HF3 HF2 HF1
				Lap_pyrs_dark.append( HFs )  # LF HF4 HF3 HF2 HF1
			# # 全局亮度调整
			# Lap_pyrs_dark[ 0 ] = self.illumination_global( img = x_dark , LF = Lap_pyrs_dark[ 0 ] ,
			# 	                                               level = self.num_high )
			#
			# # HFs向下 细节增强
			# for i in range( self.num_high ) :
			# 	# 从高频输出返回的重点信息位置
			# 	Lap_pyrs_dark[ 0 ] = self.__getattr__( 'down_guide_attention{}'.format( i ) )(
			# 			HF = Lap_pyrs_dark[ -1 - i ] , level = self.num_high - i ,
			# 			LF = Lap_pyrs_dark[ 0 ] )  # 得到LA位置注意力

			# # 亮图通路
			# Lap_pyrs_light = [ ]
			# Lap_pyrs_region = [ ]
			# # 整理输出
			# pyrs_light = self.lap_pyramid.pyramid_decom( img = x_light )  # HF1 HF2 HF3 HF4 LF
			# for i in range( self.num_high ) :
			# 	Lap_pyrs_light.append( pyrs_light[ -1 - i ] )  # LF HF4 HF3 HF2 HF1
			# # 全局亮度调整
			# Lap_pyrs_region = Lap_pyrs_light
			# Lap_pyrs_light[ 0 ] = self.illumination_global( img = x_light , LF = Lap_pyrs_light[ 0 ] ,
			#                                                 level = self.num_high )
			#
			# # 不同来源 图像重建
			# # 照明调整 Lap:LF HF4 HF3 HF2 HF1
			# Lap_illum = Lap_pyrs_dark
			# Lap_region = Lap_pyrs_dark
			# Lap_highLvl = Lap_pyrs_light
			# Lap_illum[ 0 ] = Lap_pyrs_light[ 0 ]  # HF_enhanc+LF_illum,证明根据亮度动态调整
			# Lap_region[ 0 ] = Lap_pyrs_region[ 0 ]  # HF_enhanc+LF_region,证明暗图调整后接近源域亮度
			# Lap_highLvl[ 0 ] = Lap_pyrs_dark[ 0 ]  # HF_region+LF_enhanc,证明暗图高频提取准确

			x = self.lap_pyramid.pyramid_recons( Lap_pyrs_dark )  # 输出图
			# 检测通路特征提取
			for k in range( 16 ) :# vgg13: 14 vgg16: 16
				x = self.vgg[ k ]( x )

			of1 = x
			s = self.L2Normof1( of1 )
			pal1_sources.append( s )
			# apply vgg up to fc7
			for k in range( 16,23 ) : # vgg13: 14,19 vgg16: 16,23
				x = self.vgg[ k ]( x )
			of2 = x
			s = self.L2Normof2( of2 )
			pal1_sources.append( s )

			for k in range( 23,30) :  # vgg13: 19,24 vgg16: 23,30
				x = self.vgg[ k ]( x )
			of3 = x
			s = self.L2Normof3( of3 )
			pal1_sources.append( s )

			for k in range( 30 , len( self.vgg ) ) :# vgg13: 24 vgg16: 30
				x = self.vgg[ k ]( x )
			of4 = x
			pal1_sources.append( of4 )
			# apply extra layers and cache source layer outputs

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
			self.priors_pal1 = priorbox.forward()

			priorbox = PriorBox( size , features_maps , cfg , pal = 2 )
			self.priors_pal2 = priorbox.forward()

			if self.phase == 'test' :
				output = self.detect.forward( loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) , self.softmax(
						conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) ) ,  # conf preds
				                              self.priors_pal2.type( type( x.data ) ) )

			else :
				output = (loc_pal1.view( loc_pal1.size( 0 ) , -1 , 4 ) ,
				          conf_pal1.view( conf_pal1.size( 0 ) , -1 , self.num_classes ) , self.priors_pal1 ,
				          loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) ,
				          conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) , self.priors_pal2)

		return output

	# during training, the model takes the paired images, and their pseudo GT illumination maps from the Retinex Decom Net
	def forward( self , x_dark , x_light ) :
		# x_dark输入暗图 xlight输入亮图
		size = x_dark.size()[ 2 : ]
		pal1_sources = list()
		loc_pal1 = list()
		conf_pal1 = list()
		loc_pal2 = list()
		conf_pal2 = list()

		# 金字塔分解
		pyrs_dark = self.lap_pyramid.pyramid_decom( img = x_dark )  # HF1 HF2 HF3 LF

		# 主通路(暗图)
		Lap_pyrs_dark = [ ]
		commom_guide = [ ]
		# 跨频率指导
		LF , guide = self.pipline_LF( pyrs_dark[ -1 ] )  # LF通道的两个返回值
		Lap_pyrs_dark.append( LF )  # LF
		# LF向上传递
		for i in range( self.num_high ) :
			guide = self.__getattr__( 'up_guide_layer_{}'.format( i ) )( guide )
			commom_guide.append( guide )
		# HFs完整通路
		for i in range( self.num_high ) :
			# 高频信号通路,从HF4开始
			HFs = self.__getattr__( 'pipline_HFs{}'.format( i ) )( pyrs_dark[ -2 - i ] ,
			                                                       commom_guide[ i ] )  # HF4 HF3 HF2 HF1
			Lap_pyrs_dark.append( HFs )  # LF HF3 HF2 HF1
		# 全局亮度调整
		Lap_pyrs_dark[ 0 ] = self.illumination_global(img= x_dark ,LF=Lap_pyrs_dark[ 0 ], level = self.num_high )

		# HFs向下 细节增强
		for i in range( self.num_high ) :
			# 从高频输出返回的重点信息位置
			Lap_pyrs_dark[ 0 ]=self.__getattr__( 'down_guide_attention{}'.format( i ) )(
					HF=Lap_pyrs_dark[ -1 - i ] , level=self.num_high - i ,LF=Lap_pyrs_dark[ 0 ])  # 得到LA位置注意力

		# 亮图通路
		Lap_pyrs_light = [ ]
		# Lap_pyrs_region = [ ]
		# 整理输出
		pyrs_light = self.lap_pyramid.pyramid_decom( img = x_light )  # HF1 HF2 HF3 LF
		for i in range( self.num_high+1 ) :#传递4个分频
			Lap_pyrs_light.append( pyrs_light[ -1 - i ] )  # LF HF3 HF2 HF1

		# 全局亮度调整
		# Lap_pyrs_region = Lap_pyrs_light
		Lap_pyrs_light[ 0 ] = self.illumination_global(img= x_light ,LF=Lap_pyrs_light[ 0 ], level = self.num_high )

		# 不同来源 图像重建
		# 照明调整 Lap:LF HF3 HF2 HF1
		HFs_dark = Lap_pyrs_dark[1:]
		HFs_light = Lap_pyrs_light[1:]
		loss_mutual = 0

		for i in range( len(HFs_dark) ) :
			HF_dark=HFs_dark[i].flatten( start_dim = 2 ).mean( dim = -1 )
			HF_light=HFs_light[i].flatten( start_dim = 2 ).mean( dim = -1 )
			loss_mutual = loss_mutual+cfg.WEIGHT.MC *( self.KL( HF_dark , HF_light ) + self.KL( HF_light , HF_dark ))

		# 检测通路特征提取
		x = self.lap_pyramid.pyramid_recons( Lap_pyrs_dark )  # 输出图
		for k in range( 16 ) :  # vgg13: 14 vgg16: 16
			x = self.vgg[ k ]( x )

		of1 = x  # 16个vgg后的输出
		s = self.L2Normof1( of1 )
		pal1_sources.append( s )
		# apply vgg up to fc7
		for k in range(16,23) :  # vgg13: 14,19 vgg16: 16,23
			x = self.vgg[ k ]( x )
		of2 = x
		s = self.L2Normof2( of2 )
		pal1_sources.append( s )

		for k in range( 23,30) :  # vgg13: 19,24 vgg16: 23,30
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
		self.priors_pal1 = priorbox.forward().requires_grad_(False)
		priorbox = PriorBox( size , features_maps , cfg , pal = 2 )
		self.priors_pal2 = priorbox.forward().requires_grad_(False)

		if self.phase == 'test' :
			output = self.detect.forward( loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) , self.softmax(
					conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) ) ,  # conf preds
			                              self.priors_pal2.type( type( x.data ) ) )

		else :
			output = (loc_pal1.view( loc_pal1.size( 0 ) , -1 , 4 ) ,
			          conf_pal1.view( conf_pal1.size( 0 ) , -1 , self.num_classes ) , self.priors_pal1 ,
			          loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) ,
			          conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) , self.priors_pal2)

		return output , loss_mutual

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

	def kaiming( self , param ) :
		init.kaiming_uniform( param )

	def weights_init( self , m ) :
		if isinstance( m , nn.Conv2d ) :
			self.kaiming( m.weight.data )
			m.bias.data.zero_()

		if isinstance( m , nn.ConvTranspose2d ) :
			self.kaiming( m.weight.data )
			if 'bias' in m.state_dict().keys() :
				m.bias.data.zero_()

		if isinstance( m , nn.BatchNorm2d ) :
			m.weight.data[ ... ] = 1
			m.bias.data.zero_()


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
	vgg_source = [14, 21, 28, -2]  # vgg13: [ 12 , 17 , 22 , -2 ] vgg16: [14, 21, 28, -2]

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
		super( DistillKL , self ).__init__()
		self.T = T

	def forward( self , y_s , y_t ) :
		# y_s学生模型的输出，y_t 教师模型的输出
		p_s = F.log_softmax( y_s / self.T , dim = 1 )  # 对数概率分布
		p_t = F.softmax( y_t / self.T , dim = 1 )  # 概率分布
		# 计算KL散度
		# size_average不使用平均损失，而是返回总损失，(self.T ** 2)补偿温度缩放，/ y_s.shape[0]计算平均损失
		loss = F.kl_div( p_s , p_t , size_average = False ) * (self.T ** 2) / y_s.shape[ 0 ]
		return loss
