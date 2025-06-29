"""
This file contains original functions required for the TPS transformation of bone markers, muscles and skin.
Author Ekaterina Stansfield
15 June 2025

"""


import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import json
import os
from tps import ThinPlateSpline
#import plotly.graph_objects as go
#import plotly.express as px
import pyvista as pv

from rotation_utils import rotation_matrix

# Import OSIM markers from XML file

class OsimBoneMarkers:
    def __init__(self, path_to_xml):
        self.path = path_to_xml

    def parse_xml(self) -> list:
        # parse the .xml with markers in body frames
        markers_tree=ET.parse(self.path)
        markers_root = markers_tree.getroot()
        # extract markers and create a list of dictionaries
        mrkrs_list_of_dicts = []
        for Marker in markers_root.iter('Marker'):
            dic = Marker.attrib
            for child in Marker:
                tagg = child.tag
                dic[tagg] = child.text
            mrkrs_list_of_dicts.append(dic)
        return mrkrs_list_of_dicts
    
    def data_frame(self) -> pd.DataFrame:
        mrkrs_list_of_dicts = self.parse_xml()
        # split information for data frame
        mrkrs_new = []
        keys = ['name', 'socket_parent_frame', 'location']
        for mrkr in mrkrs_list_of_dicts:
            dic = {}
            for key in keys:
                if key == 'name':
                    dic[key] = mrkr[key]
                if key == 'socket_parent_frame':
                    if '/bodyset' in mrkr[key]:
                        mrkr[key] = mrkr[key][9:]
                    elif '/' in mrkr[key]:
                        mrkr[key] = mrkr[key][1:]
                    dic['body'] = mrkr[key]
                if key == 'location':
                    a = mrkr[key].split()
                    a = [float(i) for i in a]
                    dic['r'] = a[0]#*1000
                    dic['a'] = a[1]#*1000
                    dic['s'] = a[2]#*1000
            mrkrs_new.append(dic)

        # create the dataframe that contains all information
        osim_bone_markers_df = pd.DataFrame.from_dict(mrkrs_new)
        osim_bone_markers_df.set_index('name', inplace=True)
        return osim_bone_markers_df


class OsimMusclePathsAndWrapping:
    def __init__(self, path_to_osim):
        self.path = path_to_osim
        self.muslce_paths_df()
        self.wrap_cylinder_df()

    def parse_osim(self):
        # Import and parse the original Rajagopal model with muscles
        tree_model = ET.parse(self.path)
        root_model = tree_model.getroot()
        return root_model

    def muslce_paths_df(self) -> None:
        root_model = self.parse_osim()
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
        
                a = landmark['location'].split()
                a = [float(i) for i in a]
                entry['r'] = a[0]#*1000
                entry['a'] = a[1]#*1000
                entry['s'] = a[2]#*1000
                
                body = landmark['socket_parent_frame']
                entry['body'] = body[9:]        
                m_data.append(entry)

        # create the dataframe with all muscles and their bodies
        self.df_muscles = pd.DataFrame(m_data)
        # return df_muscles
    
    def wrap_cylinder_df(self)  -> pd.DataFrame:
        root_model = self.parse_osim()
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
                half_length = 0.5*float(obj.find('length').text)
                ob['body'], ob['name'] = (body_name, obj_name)
                ob['rotation'], ob['translation'] = (rotation, translation) # rotation is in radians
                ob['radius'], ob['half_length'] = (radius, half_length)
                wrp_lst.append(ob)
        self.wrp_df = pd.DataFrame(wrp_lst)
        # return wrp_df

    
class MRIBoneMarkers:
    def __init__(self, path_to_json, orient = True):
        self.path = path_to_json
        self.orient = orient

    ## !! ATTENTION: !! 3DSlicer json separates the position and orientation in two tensors. To obtain the Viewer's orientation one needs to multiply them.
    def load_orient_json(self) -> dict:
        path_to_json = self.path
        file = open(path_to_json)
        data = json.load(file)
        points = {}
        for item in data['markups'][0]['controlPoints']:
            orientation_matrix = np.array(item['orientation'])
            orientation_matrix.shape = (3,3)
            if self.orient:
                position = np.matmul(orientation_matrix, np.array(item['position']))
            else: position = np.array(item['position'])
            points[item['label']] = position.tolist()
        return points
       
    def plot_mri_from_json(self):
        dic = self.load_orient_json(self)
        temp = pd.DataFrame.from_dict(dic).T
        fig = px.scatter(width=600, height=1000)
        fig.add_trace(go.Scatter3d(x=temp[0], y=temp[1], z=temp[2], 
                                   marker=dict(size=4, opacity=0.8),
                                   mode = 'markers + text',
                                   text=temp.index,
                                   textposition="top center"))
        fig.update_scenes(aspectmode="data" )
        fig.update_scenes(aspectmode="data" )
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2, y=2, z=2)
        )
        fig.update_layout(scene_camera=camera)
        fig.show()

    def json_to_df(self) -> pd.DataFrame:
        dic = self.load_orient_json()
        mri_bone_markers_df = pd.DataFrame.from_dict(dic, orient='index', dtype=None, columns=['r', 'a', 's'])
        return mri_bone_markers_df
    
class ScalingDF:
    def __init__(self, osim_bone_markers_df, mri_bone_markers_df):
        self.osim = osim_bone_markers_df
        self.mri = mri_bone_markers_df

    # this function defines scaling as chosen for the present project
    def lengths_to_df(self, dataframe, column_name) -> pd.DataFrame:
            # pelvis height
        ischium = (dataframe.loc['isch_tuber_r', ['r','a','s']] + dataframe.loc['isch_tuber_l', ['r','a','s']])*0.5
        ischium = ischium.to_numpy()
        ilium = (dataframe.loc['ilium_r', ['r','a','s']] + dataframe.loc['ilium_l', ['r','a','s']])*0.5
        ilium = ilium.to_numpy()
        pelvis_height = np.linalg.norm(ilium-ischium)
            # pelvis width
        pelvis_width = np.linalg.norm((dataframe.loc['femur_l_center_in_pelvis', ['r','a','s']]-dataframe.loc['femur_r_center_in_pelvis', ['r','a','s']]).to_numpy())
            # pelvis depth
        poster = (dataframe.loc['PSIS_r', ['r','a','s']] + dataframe.loc['PSIS_l', ['r','a','s']])*0.5
        anter = (dataframe.loc['ASIS_r', ['r','a','s']] + dataframe.loc['ASIS_l', ['r','a','s']])*0.5
        pelvis_depth = np.linalg.norm((poster - anter).to_numpy())
            # femur_r length
        femur_r_length = np.linalg.norm((dataframe.loc['femur_r_center', ['r','a','s']]-dataframe.loc['knee_r_center_in_femur_r', ['r','a','s']]).to_numpy())
            # femur_r width
        femur_r_width = np.linalg.norm((dataframe.loc['knee_r_med', ['r','a','s']]-dataframe.loc['knee_r_lat', ['r','a','s']]).to_numpy())
            # femur_l length
        femur_l_length = np.linalg.norm((dataframe.loc['femur_l_center', ['r','a','s']]-dataframe.loc['knee_l_center_in_femur_l', ['r','a','s']]).to_numpy())
            # femur_l width
        femur_l_width = np.linalg.norm((dataframe.loc['knee_l_med', ['r','a','s']]-dataframe.loc['knee_l_lat', ['r','a','s']]).to_numpy())
            # tibia_r length
        tibia_r_length = np.linalg.norm((dataframe.loc['tibia_r_center', ['r','a','s']]-dataframe.loc['ankle_r_center', ['r','a','s']]).to_numpy())
            # tibia_r_width
        tibia_r_width = np.linalg.norm((dataframe.loc['tibia_r_med', ['r','a','s']]-dataframe.loc['tibia_r_lat', ['r','a','s']]).to_numpy())
            # tibia_l length
        tibia_l_length = np.linalg.norm((dataframe.loc['tibia_l_center', ['r','a','s']]-dataframe.loc['ankle_l_center', ['r','a','s']]).to_numpy())
            # tibia_l_width
        tibia_l_width = np.linalg.norm((dataframe.loc['tibia_l_med', ['r','a','s']]-dataframe.loc['tibia_l_lat', ['r','a','s']]).to_numpy())
        lengths_frame_index = ['pelvis_height', 'pelvis_width', 'pelvis_depth', 'femur_r_length', 'femur_r_width', 'femur_l_length', 'femur_l_width', 'tibia_r_length', 'tibia_r_width', 'tibia_l_length', 'tibia_l_width']
        lengths_frame_list = [pelvis_height, pelvis_width, pelvis_depth, femur_r_length, femur_r_width, femur_l_length, femur_l_width, tibia_r_length, tibia_r_width, tibia_l_length, tibia_l_width]
        lengths_df = pd.DataFrame(lengths_frame_list, index=lengths_frame_index, columns=[column_name])
        return lengths_df
    
    def combine(self) -> pd.DataFrame:
        osim_df = self.lengths_to_df(self, self.osim, 'osim')
        mri_df = self.lengths_to_df(self, self.mri, 'mri')
        scaling_df = osim_df.copy()
        scaling_df['mri'] = mri_df['mri']
        scaling_df['factors'] = scaling_df['mri']/scaling_df['osim']
        return scaling_df # this needs exporting into a csv / scaling_df['factors'] = scaling_df['mri']/scaling_df['osim'] /
    

class OsimMriBoneByBodies:
    # MRI data comes not separated by bodies and without body tags
    # OSIM data comes together with body sockets
    # This class takes dataframes created earlier and matches MRI data with OSIM by bodies

    def __init__(self, osim_bone_markers_df, mri_bone_markers_df):
        self.osim = osim_bone_markers_df
        self.mri = mri_bone_markers_df
        # self.split_osim_by_bodies()
        # self.split_mri_by_osim_bodies()
        self.match_markers()


    def match_markers(self) -> tuple:
        match_markers = []
        not_in_osim = []
        for name in self.mri.index:
            if name in self.osim.index:
                match_markers.append(name)
            else: not_in_osim.append(name)
        not_in_mri = []
        for name in self.osim.index:
            if name not in self.mri.index:
                not_in_mri.append(name)
        
        self.match_markers = match_markers
        self.not_in_osim = not_in_osim
        self.not_in_mri = not_in_mri

        return (match_markers, not_in_osim, not_in_mri)


    def split_osim_by_bodies(self) -> dict:
        #match_markers = self.match_markers()[0]
        osim_bone_markers_df = self.osim.loc[self.match_markers]
        # split osim bone markers by bodies
        bodies = osim_bone_markers_df['body'].unique()
        temp = {}
        for body in bodies:
            df = osim_bone_markers_df[osim_bone_markers_df['body']==body]
            temp[body] =  df[['r', 'a', 's']] 
        return temp
    
    def split_osim_skin_by_bodies(self) -> dict:
        osim_bone_markers_df = self.osim.loc[self.not_in_mri]
        bodies = osim_bone_markers_df['body'].unique()
        temp = {}
        for body in bodies:
            df = osim_bone_markers_df[osim_bone_markers_df['body']==body]
            temp[body] =  df[['r', 'a', 's']] 
        return temp
        
    def split_mri_by_osim_bodies(self) -> tuple:
        # pelvis_osim, femur_r_osim, femur_l_osim, tibia_r_osim, tibia_l_osim, patella_r_osim, patella_l_osim = self.split_osim_by_bodies()
        # create a function for splitting the dataframe into bodies
        osim_templates_dict = self.split_osim_by_bodies()
        
        def choose_lms(mri_df, osim_template):
            points = osim_template.index
            rows = []
            for point in points:
                if point in mri_df.index:
                    rows.append({'names':point, 'r' : mri_df.loc[point, 'r'], 'a' : mri_df.loc[point, 'a'], 's' : mri_df.loc[point, 's']})
            df = pd.DataFrame.from_dict(rows)
            if df.empty != True: 
                df.set_index('names', inplace=True)
                return df       
        bodies = list(osim_templates_dict.keys())
        temp = {}        
        for body in bodies:
            df = choose_lms(self.mri, osim_templates_dict[body]) 
            if type(df) != None:
                temp[body] = df    
        return temp

# this class should take in a muscles dataframe after parsing an OSIM model
class OsimMusclesByBodies:
    def __init__(self, osim_muscle_paths_df):
        self.osim_muscles = osim_muscle_paths_df
        #self.extract_all()

    # extract muscles from the OSIM template data
    # a function for extracting muscle paths
    def extract_data(self, df, body) -> pd.DataFrame:
        rows = []
        rows.append(df[df['body']==body])
        df_rows = pd.concat(rows)[['label', 'r', 'a', 's']]
        df_new = df_rows.rename(columns={'label':'name',})
        df_new.set_index('name', inplace=True)
        return df_new

    def extract_all(self) -> None:
    # # extract muscles from the OSIM template data
        bodies = self.osim_muscles['body'].unique()
        temp = {}
        for body in bodies:
            temp[body] = self.extract_data(self.osim_muscles, body)
        return temp
"""
body = OsimMusclesByBodies(data)
body.pelvis_osim_mscl
"""
class OsimWrapsByBodies:
    def __init__(self, wrap_df):
        self.wrap_df = wrap_df

    def wraps_to_points_by_bodies(self) -> dict:
        wrp_df = self.wrap_df
        # calculate center points for the location of radius and axis of the wrapping surface
        # add them to the table
        wrp_df['radius_point']=['']*wrp_df.shape[0]
        wrp_df['axis_point']=['']*wrp_df.shape[0]
        for i, row in wrp_df.iterrows():
            axes = np.array([[1,0,0],[0,1,0],[0,0,1]])
            angles = row['rotation']
            center = row['translation']

            x_rotation=rotation_matrix(axes[0], angles[0])
            y_rotation=rotation_matrix(axes[1], angles[1])
            z_rotation=rotation_matrix(axes[2], angles[2])
            R = np.matmul(z_rotation, x_rotation, y_rotation)

            # try to create radius point in the negative X direction:
            radius_point = np.matmul(R,axes[0]*(-1))*row['radius'] + center
            axis_point = np.matmul(R,axes[2])*row['half_length']*0.5 + center
            
            wrp_df._set_value(i,'radius_point', radius_point) # = 
            wrp_df._set_value(i,'axis_point', axis_point)
        
        # create a list of bodies and split data by them
        body_lst = wrp_df['body'].unique()
        temp = {}
        for body in body_lst:
            temp[body] = wrp_df[wrp_df['body']==body][['name', 'translation','radius_point', 'axis_point']]
            # globals()[f'{body}_wrp'] = pd.concat(temp)
        return temp

        # create one list of split surfaces
        # wrapping_surface_markers = (pelvis_wrp, femur_r_wrp, femur_l_wrp, tibia_r_wrp, tibia_l_wrp)
        # return wrapping_surface_markers

    # the above function is needed to prepair wraps for tps transformation



class ImportScaleMultipleVTP:
# imports VTP files from a list and converts meters to mm
    def _init_(self, path, list_of_files):
        self.path = path
        self.list_of_files = list_of_files

    def surface_names(self):
        lst = []
        for file in self.list_of_files:
            lst.append(file[:-4])
        return lst
    
    def convert_one(self, file) -> pv.PolyData:
        mesh = pv.read(os.path.join(self.path, file))
        faces = mesh.faces
        pts = mesh.points 
        new_mesh = pv.PolyData(pts*1000, faces)
        return new_mesh
    
    def convert_all(self) -> tuple:
        names = self.surface_names()
        lst = []
        for i, file in enumerate(self.list_of_files):
            locals()[names[i]] = self.convert_one(file) 
            lst.append(locals()[names[i]])
        return tuple(lst)
    
class ImportVTP:
    def __init__(self, path, filename):
        self.path = path
        self.file = filename
        #self.name = filename[:-4]
        self.read()
        self.scale()

    def read(self) -> None:
        self.mesh = pv.read(os.path.join(self.path, self.file))

    def scale(self) -> None:
        faces = self.mesh.faces
        pts = self.mesh.points 
        self.scaled = pv.PolyData(pts*1000, faces)
        # return new_mesh
    

# TPS functions: create functions on bone markers, apply to all markers and muscles for each body

class OneBodyTPS:
    def __init__(self, body_name, 
                 osim_bone, 
                 mri_bone, 
                 osim_muscle = None,
                 osim_skin = None,  
                 exclude_bone_markers = None):
        self.name = body_name
        self.osim_bone = osim_bone
        self.mri_bone = mri_bone
        self.osim_skin = osim_skin
        self.osim_muscle = osim_muscle
        self.exclude = exclude_bone_markers

        self.define_tps_spline()
        self.apply_spline_to_bone_and_muscle()
    
    def define_tps_spline(self):
        if self.exclude != None:
            include = [x for x in self.osim_bone.index if x not in self.exclude]
        else:   include = self.osim_bone.index
            # create a spline transformation
        self.tps_spline = ThinPlateSpline(alpha = 0.02)
        self.tps_spline.fit(self.osim_bone.loc[include, ['r','a','s']].to_numpy(), 
                       self.mri_bone.loc[include, ['r','a','s']].to_numpy())
            # apply transformation to the full set of bone markers and muscle paths
 
    def apply_spline_to_bone_and_muscle(self):    
        self.transformed_bone = self.tps_spline.transform(self.osim_bone[['r','a','s']].to_numpy())
        
        if type(self.osim_muscle) == pd.core.frame.DataFrame:
            self.transformed_muscle = self.tps_spline.transform(self.osim_muscle[['r','a','s']].to_numpy())
        
        if type(self.osim_skin) == pd.core.frame.DataFrame:
            self.transformed_skin = self.tps_spline.transform(self.osim_skin[['r','a','s']].to_numpy())
        
    def apply_spline_to_surface(self, pyvista_polydata):
        pts = pyvista_polydata.points
        fcs = pyvista_polydata.faces
        surface_points = self.tps_spline.transform(np.array(pts))
        surface = pv.PolyData(surface_points, fcs)
        return surface

    def apply_tps_to_wraps(self, df) -> list:
        wraps_names = df['name']
        wraps_number = len(wraps_names)
        all_wraps_numpy = np.concatenate([np.array([i for i in df['radius_point']]), 
                                 np.array([i for i in df['axis_point']]), 
                                 np.array([i for i in df['translation']])])
        transformed_wraps = self.tps_spline.transform(all_wraps_numpy)
        [radius, axis, translation] = np.split(transformed_wraps, [wraps_number, wraps_number*2])
        return [radius, axis, translation]

#  rotation to child body
# 1) establish desired axes in osim
# 2) establish recommended axes in mri
# 3) create a rotation matrix

class TransformBodyToOsim:
    def __init__(self, mri_axes):
        self.mri_axes = mri_axes
        self.set_osim_axes()
        self.transform()
    
    def set_osim_axes(self):
        # axes positions are swapped from Slicer to Osim
        self.osim_axes = np.array([[0,0,1], [1,0,0], [0,1,0]])*50

    def transform(self): # first and second are 3x3 matrices - second is transformed to the first
        first, second = self.osim_axes, self.mri_axes
            # calculate centroids
        center_first = np.mean(first, axis = 0)
        center_second = np.mean(second, axis = 0 )
            # subtract centroids
        zeroed_first = first - center_first
        zeroed_second = second - center_second
            # find rotation using singular value decomposition
        H = np.matmul(zeroed_second.T, zeroed_first )
        U,S,Vt = np.linalg.svd(H)
        self.R = np.matmul(Vt.T, U.T)
            # check that determinate is not negative and fix if needed
        if np.linalg.det(self.R)<0:
            print('the determinant is less than zero, recalculate r')
            Vt[2,:] *= -1
            self.R =np.matmul(Vt.T, U.T)
        return self.R
            # define tranlsation
        # new_center_second = np.mean(np.matmul(R,zeroed_second.T), axis = 0)
        # t = -new_center_second + center_first  
        # return R, t #, center_first, center_second


    
class GetPelvisAxes(TransformBodyToOsim):
    def __init__(self, bone_numpy, bone_markers):
        self.bone_markers = bone_markers
        self.bone_numpy = bone_numpy
        self.define_mri_axes()
        TransformBodyToOsim.__init__(self, self.mri_axes)
        self.apply_to_bone()

    
    def plane_normal(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        v3 = np.cross(v1, v2)
        return v3/np.linalg.norm(v3)

    def define_mri_axes(self) -> None: 
        pub_super_c = ''
        asis_r = ''
        asis_l = ''    
        for i, marker in enumerate(self.bone_markers):
            if marker == 'pub_super_c':
                pub_super_c = self.bone_numpy[i]
            if marker == 'ASIS_r':
                asis_r = self.bone_numpy[i]
            if marker == 'ASIS_l':
                asis_l = self.bone_numpy[i]
        
        pelvis_origin = np.mean([asis_r, asis_l], axis = 0)
        
        # lr stands for left to right
        pelvis_lr = (asis_r - asis_l)/np.linalg.norm(asis_r - asis_l)
        # py stands for posterior to anterior
        pelvis_pa = self.plane_normal(asis_l,  pub_super_c, asis_r)
        # is stands for inferior -> superior
        pelvis_is = np.cross(pelvis_lr, pelvis_pa)
        self.mri_axes = (np.array((pelvis_lr, pelvis_pa, pelvis_is)) * 50) + pelvis_origin 
 

    # rotate and shift origin to 0,0,0
    def apply_to_bone(self):
        data_zeroed = self.bone_numpy - np.mean(self.bone_numpy, axis = 0)
        rotated = (np.matmul(self.R, data_zeroed.T)).T
        for i, marker in enumerate(self.bone_markers):
            if marker == 'ASIS_r':
                asis_r = rotated[i]
            if marker == 'ASIS_l':
                asis_l = rotated[i]
        self.pelvis_origin = np.mean([asis_r, asis_l], axis = 0)
        # pelvis_origin = np.mean(rotated[:2], axis = 0) # mean of ASIS potins
        self.bone_transformed = rotated - self.pelvis_origin

    def apply_to_non_bone(self, data_numpy):
        data_zeroed = data_numpy - np.mean(self.bone_numpy, axis = 0)
        rotated = (np.matmul(self.R, data_zeroed.T)).T
        pelvis_transformed = rotated - self.pelvis_origin
        return pelvis_transformed
        
class GetFemurAxes(TransformBodyToOsim):
    def __init__(self, femur_bone_numpy, femur_bone_markers,  femur_skin_numpy = None, femur_muscles_numpy = None, femur_wraps_numpy = None, femur_surface_numpy = None):
        self.bone_markers = femur_bone_markers
        self.bone_numpy = femur_bone_numpy
        self.skin_numpy = femur_skin_numpy
        self.muscles_numpy = femur_muscles_numpy
        self.wraps_numpy = femur_wraps_numpy
        self.surface_numpy = femur_surface_numpy
        self.define_side()
        self.define_mri_axes()
        TransformBodyToOsim.__init__(self, self.mri_axes)
        # self.bone_data_center()
        self.apply_to_bone()
        self.transform_non_bone()
        #self.transform_patella()

    def define_side(self) -> None:
        if 'femur_r_center' in self.bone_markers:
            self.side = 'r'
        if 'femur_l_center' in self.bone_markers:
            self.side = 'l'
        #else: print('Marker names do not correspond with expected')

    def plane_normal(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        v3 = np.cross(v1, v2)
        return v3/np.linalg.norm(v3)

    def define_mri_axes(self) -> None: 
        for i, marker in enumerate(self.bone_markers):       
            if marker == f'femur_{self.side}_center':
                femur_head_center = self.bone_numpy[i]
            if marker == f'knee_{self.side}_med':
                meidal_knee= self.bone_numpy[i]
            if marker == f'knee_{self.side}_lat':
                lateral_knee = self.bone_numpy[i]
        knee_center = np.mean([meidal_knee, lateral_knee], axis = 0)
        if self.side == 'r':        
            self.femur_pa = self.plane_normal(femur_head_center, meidal_knee, lateral_knee)
        if self.side == 'l':        
            self.femur_pa = self.plane_normal(femur_head_center,  lateral_knee, meidal_knee) 
        self.femur_is = (femur_head_center - knee_center)/np.linalg.norm(femur_head_center - knee_center)
        self.femur_lr = np.cross(self.femur_pa, self.femur_is)/np.linalg.norm(np.cross(self.femur_pa, self.femur_is))
        self.mri_axes = np.array((self.femur_lr, self.femur_pa, self.femur_is)) * 50
        TransformBodyToOsim.__init__(self, self.mri_axes)

    # rotate and shift origin to 0,0,0
    def apply_to_bone(self):
        self.bone_data_center = np.mean(self.bone_numpy, axis = 0)
        
        data_zeroed = self.bone_numpy - self.bone_data_center
        
        rotated = (np.matmul(self.R, data_zeroed.T)).T
        
        for i, marker in enumerate(self.bone_markers):
            if marker == f'femur_{self.side}_center':
                femur_head_center = rotated[i]
        
        self.bone_transformed = rotated - femur_head_center
        self.non_bone_translation = femur_head_center

    def apply_to_non_bone(self, data_numpy):
        data_zeroed = data_numpy - self.bone_data_center
        rotated = (np.matmul(self.R, data_zeroed.T)).T
        transformed_to_femur = rotated - self.non_bone_translation
        return transformed_to_femur
    
    def transform_non_bone(self):
        if self.skin_numpy.any() != None:
            self.femur_skin = self.apply_to_non_bone(self.skin_numpy)
        if self.muscles_numpy.any() != None:
            self.femur_muscles = self.apply_to_non_bone(self.muscles_numpy)
        if self.surface_numpy.any() != None:
            self.femur_surface = self.apply_to_non_bone(self.surface_numpy)
        if self.wraps_numpy.any() != None:
            self.femur_wraps = self.apply_to_non_bone(self.wraps_numpy)
    
    def transform_patella(self, patella_bone_numpy, patella_bone_markers, patella_muscles_numpy=None, patella_surface_numpy = None):
        # refresh mediolateral axis
        for i, marker in enumerate(patella_bone_markers):
            if marker == f'patella_{self.side}':
                patella_location_index = i
            if marker == f'patella_lat_{self.side}':
                patella_lat_index = i
            if marker == f'patella_med_{self.side}':
                patella_med_index = i
        if self.side == 'r':
            patella_lr = patella_bone_numpy[patella_lat_index] - patella_bone_numpy[patella_med_index] 
        if self.side == 'l':
            patella_lr = patella_bone_numpy[patella_med_index] - patella_bone_numpy[patella_lat_index]
        patella_lr = patella_lr /np.linalg.norm(patella_lr)         
        patella_pa = np.cross(self.femur_is, patella_lr)/np.linalg.norm(np.cross(self.femur_is, patella_lr))
        self.mri_axes = np.array((patella_lr,  patella_pa, self.femur_is)) * 50
        TransformBodyToOsim.__init__(self, self.mri_axes)

        # apply transformation
        bone_numpy_transformed = self.apply_to_non_bone(patella_bone_numpy)        
        patella_bone = bone_numpy_transformed - bone_numpy_transformed[patella_location_index]      
        if patella_muscles_numpy.any():
            muscles_numpy_transformed = self.apply_to_non_bone(patella_muscles_numpy)
            patella_muscles = muscles_numpy_transformed- bone_numpy_transformed[patella_location_index]
        if patella_surface_numpy.any():
            surface_numpy_transformed = self.apply_to_non_bone(patella_surface_numpy)
            patella_surface = surface_numpy_transformed - bone_numpy_transformed[patella_location_index]
        return patella_bone, patella_muscles, patella_surface

    
class GetTibiaAxes(TransformBodyToOsim):
    def __init__(self, bone_markers = None, bone_numpy = None):
        self.bone_markers = bone_markers
        self.bone_numpy = bone_numpy
        self.define_side()
        self.define_mri_axes()
        TransformBodyToOsim.__init__(self, self.mri_axes)
        self.bone_data_center()
        self.tibia_points()
        self.apply_to_bone()

    def define_side(self) -> None:
        if 'knee_r_center' in self.bone_markers:
            self.side = 'r'
        if 'knee_l_center' in self.bone_markers:
            self.side = 'l'
        #else: print('Markers name do not correspond with expected')

    def plane_normal(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        v3 = np.cross(v1, v2)
        return v3/np.linalg.norm(v3)

    def define_mri_axes(self) -> None: 
        for i, marker in enumerate(self.bone_markers):
            if marker == f'tibia_{self.side}_center':
                tibia_center = self.bone_numpy[i]
            if marker == f'tibia_{self.side}_med':
                tibia_med= self.bone_numpy[i]
            if marker == f'talus_{self.side}_center_in_tibia':
                talus_center = self.bone_numpy[i]
            if marker == f'tibia_{self.side}_lat':
                tibia_lat = self.bone_numpy[i]
        if self.side == 'r':
            tibia_pa = self.plane_normal(tibia_med, talus_center, tibia_lat)
        if self.side == 'l':
            tibia_pa = self.plane_normal(tibia_lat, talus_center, tibia_med)
        tibia_is = (tibia_center - talus_center)/np.linalg.norm(tibia_center - talus_center)
        tibia_lr = np.cross(tibia_pa, tibia_is)/np.linalg.norm(np.cross(tibia_pa, tibia_is))
        self.mri_axes = np.array((tibia_lr, tibia_pa, tibia_is)) * 50 + tibia_center

    def bone_data_center(self):
        self.bone_data_center = np.mean(self.bone_numpy, axis = 0)

    def tibia_points(self):
        for i, marker in enumerate(self.bone_markers):
            if marker == f'tibia_{self.side}_center':
                self.tibia_index = i
            if marker == f'knee_{self.side}_center':
                self.knee_index = i

    def apply_to_bone(self):
        data_zeroed = self.bone_numpy - np.mean(self.bone_numpy, axis = 0)
        rotated = (np.matmul(self.R, data_zeroed.T)).T
        transformed_to_tibia_center = rotated - rotated[self.tibia_index]
        #print(transformed_to_tibia_center)
        transformed_to_tibia_center[self.knee_index] = transformed_to_tibia_center[self.knee_index]*np.array([0,1,0])
        self.bone_transformed = transformed_to_tibia_center - transformed_to_tibia_center[self.knee_index]
        self.non_bone_translation = rotated[self.tibia_index] + transformed_to_tibia_center[self.knee_index]
        #return transformed      

    # rotate and shift origin to 0,0,0
    def apply_to_not_bone(self, data_numpy):
        data_zeroed = data_numpy - self.bone_data_center
        rotated = (np.matmul(self.R, data_zeroed.T)).T
        translformed = rotated - self.non_bone_translation
        return translformed
 

class ScaleAndRecordData:
    def __init__ (self, body, output_path,
                  transformed_bone_markers = None, bone_marker_names = None,
                  transformed_skin_markers = None, skin_marker_names = None,
                  transformed_muscle_markers = None, muscle_marker_names = None,
                  transformed_wrap_translations = None, wrap_names = None,
                  list_transformed_surfaces_as_PolyData = None, list_surface_names = None):
        self.body = body
        self.output_path = output_path
        self.bone = transformed_bone_markers
        self.bone_names = bone_marker_names
        self.skin = transformed_skin_markers
        self.skin_names = skin_marker_names
        self.muscle = transformed_muscle_markers
        self.muscle_marker_names = muscle_marker_names
        self.wraps = transformed_wrap_translations
        self.wrap_names = wrap_names
        self.surfaces = list_transformed_surfaces_as_PolyData
        self.surface_names = list_surface_names
        self.set_printoptions()
        self.record_surfaces()
        self.record_bone_markers()
        if type(transformed_skin_markers) == np.ndarray:
            self.record_skin_markers()
        if type(transformed_muscle_markers) == np.ndarray:
            self.record_muscle_paths()
        if type(transformed_wrap_translations) == np.ndarray:
            self.record_wrap_translations()

    def set_printoptions(self):
        np.set_printoptions(suppress=True, precision=17)

    def record_surfaces(self):
        if len(self.surfaces) >= 1:
            if len(self.surface_names)  == len(self.surfaces):
                for i, name in enumerate(self.surface_names):
                    surface = self.surfaces[i]
                    mesh = pv.PolyData(surface.points/1000, surface.faces)
                    mesh.save(os.path.join(self.output_path, f'{name}.stl'))

    def record_bone_markers(self):
        if len(self.bone) >= 1:
            body = self.body
            diction = {'body':[], 'name':[], 'location':[]}
            for i, name in enumerate(self.bone_names):
                diction['body'].append(body),
                diction['name'].append(name),
                diction['location'].append(self.bone[i]/1000)
            df = pd.DataFrame(diction)
            df.to_csv(os.path.join(self.output_path, f'{body}_bone_markers.csv'))


    def record_skin_markers(self):
        if len(self.skin) >= 1:
            body = self.body
            diction = {'body':[], 'name':[], 'location':[]}
            for i, name in enumerate(self.skin_names):
                diction['body'].append(body),
                diction['name'].append(name),
                diction['location'].append(self.skin[i]/1000)
            df = pd.DataFrame(diction)
            df.to_csv(os.path.join(self.output_path, f'{body}_skin_markers.csv'))

    def record_muscle_paths(self):
        if len(self.muscle) >= 1:
            body = self.body
            diction = {'body':[], 'name':[], 'location':[]}
            for i, name in enumerate(self.muscle_marker_names):
                diction['body'].append(body),
                diction['name'].append(name),
                diction['location'].append(self.muscle[i]/1000)
            df = pd.DataFrame(diction)
            df.to_csv(os.path.join(self.output_path, f'{body}_muscle_paths.csv'))

    def record_wrap_translations(self):
        if len(self.wraps) >= 1:
            body = self.body
            diction = {'body':[], 'name':[], 'location':[]}
            for i, name in enumerate(self.wrap_names):
                diction['body'].append(body),
                diction['name'].append(name),
                diction['location'].append(self.wraps[i]/1000)
            df = pd.DataFrame(diction)
            df.to_csv(os.path.join(self.output_path, f'{body}_wrap_translations.csv'))

