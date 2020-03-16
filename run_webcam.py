import argparse
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils import video


import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import ast

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture
from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

import sys
#sys.path.append('/Users/stevenye/cv/face3d_old')
import face3d
from face3d import mesh
from face3d import mesh_numpy
from face3d.morphable_model import MorphabelModel

CROP_SIZE = 128
DOWNSAMPLE_RATIO = 1

def resize(image):
    """Crop and resize image for pix2pix."""
    height, width, _ = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize

def main():


    # OpenCV
    #cap = cv2.VideoCapture(args.video_source)
    cap = cv2.VideoCapture('b.mov')
    fps = video.FPS().start()

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)

    #while True:
    while cap.isOpened():
        ret, frame = cap.read()

        # resize image and detect face
        frame_resize = cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)

        # read image
        image = frame_resize
        image = resize(image)

        [h, w, c] = image.shape
        if c>3:
            image = image[:,:,:3]

        # the core: regress position map
        if args.isDlib:
            max_size = max(image.shape[0], image.shape[1])
            if max_size> 1000:
                image = rescale(image, 1000./max_size)
                image = (image*255).astype(np.uint8)
            st = time()
            pos = prn.process(image) # use dlib to detect face
            print('process',time()-st)
        else:
            if image.shape[0] == image.shape[1]:
                image = resize(image, (256,256))
                pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
            else:
                box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
                pos = prn.process(image, box)
        
        image = image/255.
        if pos is None:
            cv2.imshow('a',frame_resize)
            fps.update()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if args.is3d or args.isMat or args.isPose or args.isShow:
            # 3D vertices
            vertices = prn.get_vertices(pos)
            if args.isFront:
                save_vertices = frontalize(vertices)
            else:
                save_vertices = vertices.copy()
            save_vertices[:,1] = h - 1 - save_vertices[:,1]
            #colors = prn.get_colors(image, vertices)
            #write_obj_with_colors(os.path.join('', 'webcam' + '.obj'), save_vertices, prn.triangles, colors)
        #if args.is3d:
        #    # corresponding colors
        #    colors = prn.get_colors(image, vertices)
#
        #    if args.isTexture:
        #        if args.texture_size != 256:
        #            pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range = True)
        #        else:
        #            pos_interpolated = pos.copy()
        #        texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        #        if args.isMask:
        #            vertices_vis = get_visibility(vertices, prn.triangles, h, w)
        #            uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
        #            uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range = True)
        #            texture = texture*uv_mask[:,:,np.newaxis]
        #        #write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
        #    else:
        #        True
        #        #write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)
#
        #if args.isDepth:
        #    depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
        #    depth = get_depth_image(vertices, prn.triangles, h, w)
        #    #imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)
        #    #sio.savemat(os.path.join(save_folder, name + '_depth.mat'), {'depth':depth})
#
        #if args.isKpt or args.isShow:
        #    # get landmarks
        #    kpt = prn.get_landmarks(pos)
        #    #np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)
#
        #if args.isPose or args.isShow:
        #    # estimate pose
        #    camera_matrix, pose = estimate_pose(vertices)

        #write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, colors)
        
        rendering_cc = mesh.render.render_grid(save_vertices, prn.triangles, 900, 900)
        a = np.transpose(rendering_cc,axes=[1,0,2])
        dim = rendering_cc.shape[0]

        i_t = np.ones([dim,dim,3],dtype= np.float32)
        for i in range(dim):
            i_t[i] = a[dim-1-i]
        i_t = i_t/255
        #imsave('webcam.png', i_t)

        #kpt = prn.get_landmarks(pos)

        #cv2.imshow('frame', image)
        #cv2.imshow('a',i_t/255)

        #cv2.imshow('sparse alignment', np.concatenate([image, i_t], axis=1))
        cv2.imshow('sparse alignment', i_t)
        cv2.imshow('vedio', image)
        #cv2.imshow('sparse alignment', np.concatenate([plot_kpt(image, kpt), i_t], axis=1))
        #cv2.imshow('dense alignment', plot_vertices(image, vertices))
        #cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))

        
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default=True, type=ast.literal_eval,
                        help='whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default=False, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default=False, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--isPose', default=False, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isShow', default=False, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=False, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isDepth', default=False, type=ast.literal_eval,
                        help='whether to output depth image')
    # update in 2017/4/27
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    parser.add_argument('--isMask', default=False, type=ast.literal_eval,
                        help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
    # update in 2017/7/19
    parser.add_argument('--texture_size', default=256, type=int,
                        help='size of texture map, default is 256. need isTexture is True')

    args = parser.parse_args()

    main()
