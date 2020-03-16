'''
functions about rendering mesh(from 3d obj to 2d image).
only use rasterization render here.
Note that:
1. Generally, render func includes camera, light, raterize. Here no camera and light(I write these in other files)
2. Generally, the input vertices are normalized to [-1,1] and cetered on [0, 0]. (in world space)
   Here, the vertices are using image coords, which centers on [w/2, h/2] with the y-axis pointing to oppisite direction.
 Means: render here only conducts interpolation.(I just want to make the input flexible)

Author: Yao Feng 
Mail: yaofeng1995@gmail.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time

from .cython import mesh_core_cython

def rasterize_triangles(vertices, triangles, h, w):
    ''' 
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        h: height
        w: width
    Returns:
        depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle). 
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.

    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # h, w is the size of rendering
    '''

    # initial 
    #depth_buffer = np.zeros([h, w],dtype = np.float32) - 999999. #set the initial z to the farest position
    depth_buffer = np.zeros([h, w], dtype = np.float32, order = 'C') - 999999.
    triangle_buffer = np.zeros([h, w], dtype = np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = np.zeros([h, w, 3], dtype = np.float32)  # 
    
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()

    mesh_core_cython.rasterize_triangles_core(
                vertices, 
                triangles,
                depth_buffer, 
                triangle_buffer, 
                barycentric_weight, 
                vertices.shape[0],
                triangles.shape[0], 
                h, w)

def render_colors(vertices, triangles, colors, h, w, c = 3, BG = None):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width  
        c: channel
        BG: background image
    Returns:
        image: [h, w, c]. rendered image./rendering.
    '''

    # initial 
    if BG is None:
        image = np.zeros((h, w, c), dtype = np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG
    depth_buffer = np.zeros([h, w], dtype = np.float32, order = 'C') - 999999.

    # change orders. --> C-contiguous order(column major)
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    colors = colors.astype(np.float32).copy()
    ###
    mesh_core_cython.render_colors_core(
                image, vertices, triangles,
                colors,
                depth_buffer,
                vertices.shape[0], triangles.shape[0], 
                h, w, c)
    return image



def lookat(eye, center, up):
    normalize = lambda x: x/np.linalg.norm(x)
    M = np.eye(4)
    z = normalize(eye-center)
    x = normalize(np.cross(up,z))
    y = normalize(np.cross(z,x))
    M[0,:3], M[1,:3], M[2,:3], M[:3,3] = x, y, z, -center
    return M

def viewport_fn(x, y, w, h, d):
    return np.array([[w/2, 0, 0, x+w/2],
                     [0, h/2, 0, y+h/2],
                     [0, 0, d/2,   d/2],
                     [0, 0, 0,       1]])

def render_grid(vertices, triangles, h, w, c = 3):
    '''
    vertices: [3, nver]
    triangles: [3, ntri]
    h: height of rendering
    w: width of rendering
    c: channel
    '''

    zbuffer = np.zeros([h, w], dtype = np.float32, order = 'C') - 1000.

    coords = np.mgrid[0:w, 0:h].astype(np.int32)

    image = np.zeros((h, w, c), dtype = np.float32)
    depth_buffer = np.zeros([h, w], dtype = np.float32, order = 'C') - 999999.
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    V = vertices.copy()
    Vi = triangles.copy()

    light         = np.array([0,0,-1])
    eye           = np.array([-1,1,3])
    center        = np.array([0,0,0])
    up            = np.array([0,1,0])
    # Centering and scaling
    #vmin, vmax = V.min(), V.max()
    #V = (2*(V-vmin)/(vmax-vmin) - 1)*1.25
    for i in range(3):
        vmin, vmax = V[:,i].min(),V[:,i].max()
        V[:,i] = (2*(V[:,i]-vmin)/(vmax-vmin) - 1)*0.8
        
    xmin, xmax = V[:,0].min(), V[:,0].max()
    V[:,0] = V[:,0] - xmin - (xmax-xmin)/2
    ymin, ymax = V[:,1].min(), V[:,1].max()
    V[:,1] = V[:,1] - ymin - (ymax-ymin)/2
    
    viewport = viewport_fn(32, 32, w-64, h-64, 1000)
    modelview = lookat(eye, center, up)
    
    Vh = np.c_[V, np.ones(len(V))] # Homogenous coordinates
    V = Vh @ modelview.T           # World coordinates
    Vs = V @ viewport.T            # Screen coordinates
    Vs = Vs.astype(np.float32)
    V, Vs = V[:,:3],  Vs[:,:3]     # Back to cartesian coordinates
    
    V, Vs = V[Vi], Vs[Vi]
    
    # Pre-compute tri-linear coordinates
    T = np.transpose(Vs, axes=[0,2,1]).copy()
    T[:,2,:] = 1
    #T = np.linalg.inv(T)
    T = T.astype(np.float32)
    
    # Pre-compute normal vectors and intensity
    N = np.cross(V[:,2]-V[:,0], V[:,1]-V[:,0])
    N = N / np.linalg.norm(N,axis=1).reshape(len(N),1)
    I = np.dot(N, light)*255
    I = I.astype(np.float32)

    mesh_core_cython.render_grid_core(
                image, T, Vs, I, zbuffer, coords, vertices, triangles,
                depth_buffer,
                vertices.shape[0], triangles.shape[0],
                h, w, c)
    #print('----------call c++ render_grid_core 秒数: ', time() - st)
    return image


def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, h, w, c = 3, mapping_type = 'nearest', BG = None):
    ''' render mesh with texture map
    Args:
        vertices: [3, nver]
        triangles: [3, ntri]
        texture: [tex_h, tex_w, 3]
        tex_coords: [ntexcoords, 3]
        tex_triangles: [ntri, 3]
        h: height of rendering
        w: width of rendering
        c: channel
        mapping_type: 'bilinear' or 'nearest'
    '''
    # initial 
    if BG is None:
        image = np.zeros((h, w, c), dtype = np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG

    depth_buffer = np.zeros([h, w], dtype = np.float32, order = 'C') - 999999.
    
    tex_h, tex_w, tex_c = texture.shape
    if mapping_type == 'nearest':
        mt = int(0)
    elif mapping_type == 'bilinear':
        mt = int(1)
    else:
        mt = int(0)
    
    # -> C order
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    texture = texture.astype(np.float32).copy()
    tex_coords = tex_coords.astype(np.float32).copy()
    tex_triangles = tex_triangles.astype(np.int32).copy()

    mesh_core_cython.render_texture_core(
                image, vertices, triangles,
                texture, tex_coords, tex_triangles,
                depth_buffer,
                vertices.shape[0], tex_coords.shape[0], triangles.shape[0], 
                h, w, c,
                tex_h, tex_w, tex_c,
                mt)
    return image

