
"""
This file contains original functions required for controlling the size of wrapping surfaces.
Author Ekaterina Stansfield
15 June 2025

"""

import opensim as osim
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import math as m
from rotation_utils import point_distance_to_vector, rotation_matrix

# function to extract muscle paths, wrapping surfaces info and joint info from an osim model
# returns three dataframes: muscles, surfaces and joints

class FixWraps:
    def __init__(self, model_path):
        self.model = model_path
        self.get_model_info()
        
    def get_model_info(self):
        tree_model = ET.parse(self.model)
        root_model = tree_model.getroot()

        # Extract the muscle path points
        lst = []
        for PathPointSet in root_model.iter('PathPointSet'):
            lst.append(PathPointSet)

        lm_groups = []
        for N in lst:
            n = N[0]
            l = []
            for el in n:
                dic = el.attrib
                for child in el:
                    tagg = child.tag
                    dic[tagg] = child.text
                l.append(dic)
            lm_groups.append(l)

        # Split information in preparation for a dataframe
        m_data = []
        for muscle in lm_groups:
            for landmark in muscle:
                entry = {}
                lm = landmark['name']
                entry['label'] = lm

                msk = lm[:-3]
                entry['muscle'] = msk

                body = landmark['socket_parent_frame']
                entry['body'] = body[9:]        
                m_data.append(entry)
        
                a = landmark['location'].split()
                a = [float(i) for i in a]
                entry['r'] = a[0]
                entry['a'] = a[1]
                entry['s'] = a[2]
                
        # create the dataframe with all muscles and their bodies
        self.df_muscles = pd.DataFrame(m_data)

        # Extract WrapCylinders information from body class and from muscle class
        wrp_lst = []
        for Body in root_model.iter('Body'):
            body_name = Body.get('name')
            objs = Body.iter('WrapCylinder')
            # print(objs)
            for obj in objs:
                ob = {}
                obj_name = obj.get('name')
                rotation = (obj.find('xyz_body_rotation').text).split()
                rotation = np.array([float(i) for i in rotation])
                translation = (obj.find('translation').text).split()
                translation = np.array([float(i) for i in translation])
                radius = float(obj.find('radius').text)
                length = float(obj.find('length').text)
                ob['name'],ob['body'] = (obj_name, body_name)
                ob['rotation'], ob['translation'] = (rotation, translation) # rotation is in radians
                ob['radius'], ob['length'] = (radius, length)
                wrp_lst.append(ob)

        self.wrp_df = pd.DataFrame(wrp_lst)
        self.wrp_df.set_index('name', inplace=True)
        
        wrp_info = {}
        for mscl in root_model.iter('Millard2012EquilibriumMuscle'):
            mscl_name = mscl.get('name')
            mscl_wrp = mscl.iter('PathWrap')
            for obj in mscl_wrp:
                obj_name = obj.find('wrap_object').text
                method = obj.find('method').text
                range = obj.find('range').text
                wrp_info[obj_name] = [mscl_name, method, range]
                
        for obj in self.wrp_df.index:
            if obj in wrp_info.keys():
                self.wrp_df.loc[obj, 'muscle'] = wrp_info[obj][0]
                self.wrp_df.loc[obj,'range'] = wrp_info[obj][2]
            else: print(obj, 'is NOT present')

        # extract information about joints
        joints_info = []
        for joint in root_model.iter('CustomJoint'):
            info = {}
            name = joint.attrib['name']
            socket_parent_frame = joint.find('socket_parent_frame').text
            info['name'] = name
            for frame in joint.iter('PhysicalOffsetFrame'):
                frame_name = frame.attrib['name']
                if frame_name == socket_parent_frame:
                    info['body'] = frame.find('socket_parent').text[9:]
                    info['translation'] = frame.find('translation').text
                    info['rotation'] = frame.find('orientation').text
            joints_info.append(info)
        self.joints_df = pd.DataFrame(joints_info)

    def collect_wrp_details(self):
        wrp_adjust = {}

        for wrp_name in self.wrp_df.index:
            rot=np.array([float(x) for x in self.wrp_df.loc[wrp_name, 'rotation']])
            transl=np.array([float(x) for x in self.wrp_df.loc[wrp_name, 'translation']])
            mskl = self.wrp_df.loc[wrp_name, 'muscle']
            rad = float(self.wrp_df.loc[wrp_name, 'radius'])
            body = self.wrp_df.loc[wrp_name, 'body']

            res = self.find_new_wrp_radius(body, rot, transl, mskl, rad)
            
            if len(res) > 0:
                wrp_adjust[wrp_name] = res
        return wrp_adjust

    def find_new_wrp_radius(self, body, rot, transl, mskl, rad):
        mskl_data = self.df_muscles[self.df_muscles['muscle'].isin([mskl])]
        rad = float(rad)
        new_rad=rad
        result = ''

        for i, pt in mskl_data.iterrows():
            point = pt[['r','a','s']].to_numpy()
            point = point.astype(np.float64)
            vect = self.create_rot_vector(rot)
            point_body = pt['body']
            point_label = pt['label']
            
            if point_body == body:
                joint_location = [0,0,0]

            elif point_body == 'femur_r' and body == 'pelvis':
                lst = self.joints_df[self.joints_df['name'] == 'hip_r']['translation'].to_string().split()
                joint_location = np.array([float(x) for x in lst])[1:]
            
            elif point_body == 'femur_l' and body == 'pelvis':
                lst = self.joints_df[self.joints_df['name'] == 'hip_l']['translation'].to_string().split()
                joint_location = np.array([float(x) for x in lst])[1:]

            elif point_body == 'pelvis' and body == 'femur_r':
                lst = self.joints_df[self.joints_df['name'] == 'hip_r']['translation'].to_string().split()
                joint_location = np.array([float(x) for x in lst])[1:]
            
            elif point_body == 'pelvis' and body == 'femur_l':
                lst = self.joints_df[self.joints_df['name'] == 'hip_l']['translation'].to_string().split()
                joint_location = np.array([float(x) for x in lst])[1:]

            elif (point_body == 'patella_r' or point_body == 'tibia_r') and body == 'femur_r':
                lst = self.joints_df[self.joints_df['name'] == 'walker_knee_r']['translation'].to_string().split()
                joint_location =  - np.array([float(x) for x in lst])[1:]

            elif (point_body == 'patella_l' or point_body == 'tibia_l') and body == 'femur_l':
                lst = self.joints_df[self.joints_df['name'] == 'walker_knee_l']['translation'].to_string().split()
                joint_location = - np.array([float(x) for x in lst])[1:]

            elif point_body == 'femur_r' and body == 'tibia_r':
                lst = self.joints_df[self.joints_df['name'] == 'walker_knee_r']['translation'].to_string().split()
                joint_location = np.array([float(x) for x in lst])[1:]
            
            elif point_body == 'femur_l' and body == 'tibia_l':
                lst = self.joints_df[self.joints_df['name'] == 'walker_knee_l']['translation'].to_string().split()
                joint_location = np.array([float(x) for x in lst])[1:]

            elif point_body == 'tibia_r' and body == 'femur_r':
                lst = self.joints_df[self.joints_df['name'] == 'walker_knee_r']['translation'].to_string().split()
                joint_location = - np.array([float(x) for x in lst])[1:]
            
            elif point_body == 'tibia_l' and body == 'femur_l':
                lst = self.joints_df[self.joints_df['name'] == 'walker_knee_l']['translation'].to_string().split()
                joint_location = - np.array([float(x) for x in lst])[1:]
            else:
                continue

            dist, dist_to_transl = self.via_point_distance(point, vect, rad, transl, body, point_body, point_label, joint = joint_location)
            
            if (float(dist_to_transl) <= rad) or (float(dist) <= rad):
                # adjustment = round(rad - dist, 3)
                # new_rad = rad - adjustment
                result = [body, transl, rad, point_label, point, point_body, joint_location, joint_location+point, dist_to_transl,  dist] #, adjustment, new_rad

        return result
    
    def via_point_distance(self, point, vect, rad, transl, body, point_body, point_label, joint = [0,0,0]):
        jt_pt = - np.array(joint) + np.array(point)
        vt_tr = np.array(vect) #+ transl
        dist = point_distance_to_vector(jt_pt - transl,  vt_tr) # point_distance_to_vector(joint + point - transl, vect + transl)
        dist_to_transl = np.linalg.norm(jt_pt - transl)
        return (dist, dist_to_transl)

    def create_rot_vector(self, rot):
        axes = np.array([[1,0,0],[0,1,0],[0,0,1]])
        x_rotation=rotation_matrix(axes[0], rot[0])
        y_rotation=rotation_matrix(axes[1], rot[1])
        z_rotation=rotation_matrix(axes[2], rot[2])
        R = x_rotation + y_rotation + z_rotation
        vect = np.matmul(R,np.array( [0,0,1]))
        vect = vect/np.linalg.norm(vect)
        return vect
                
        

