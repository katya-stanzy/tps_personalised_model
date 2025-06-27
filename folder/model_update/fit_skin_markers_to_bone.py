import numpy as np
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


class FitSkinMarkersToBone():
    def __init__(self, transformed_bone_markers_df, path_to_static, model_to_update, updated_model):
        self.markers_df = transformed_bone_markers_df
        self.path_to_static = path_to_static
        self.model_to_update = model_to_update
        self.updated_model = updated_model
        self.split_markers_df_by_bodies()
        self.import_stat_experim_markers()
        self.homologous_pairs()

    def transforms(self, first, second): # second set of makers is transformed to the first
            # calculate centroids
        center_first = np.mean(first, axis = 0)
        center_second = np.mean(second, axis = 0 )
            # subtract centroids
        zeroed_first = first - center_first
        zeroed_second = second - center_second
            # find rotation using singular value decomposition
        H = np.matmul(zeroed_second.T, zeroed_first )
        U,S,Vt = np.linalg.svd(H)
        R = np.matmul(Vt.T, U.T)  
            # check that determinate is not negative and fix if needed
        if np.linalg.det(R)<0:
            print('the determinant is less than zero, recalculate r')
            Vt[2,:] *= -1
            R =np.matmul(Vt.T, U.T)
            # define tranlsation
        new_center_second = np.mean(np.matmul(R, zeroed_second.T).T, axis = 0)
        t = -new_center_second + center_first  
        return R, center_second, t

    def return_dataframe(self, body):
        extract_df = self.markers_df[self.markers_df['body']== body]['location']
        ind = extract_df.index
        col = ['x', 'y', 'z']
        list_of_data = [row[1:-1].split() for row in extract_df]
        convert = lambda x: 0 if x == '.' else float(x)
        data = np.array([[convert(x) for x in row] for row in list_of_data])
        data_df = pd.DataFrame(data, index=ind, columns=col)
        return data_df

    def split_markers_df_by_bodies(self):
        self.pelvis_df = self.return_dataframe('pelvis')
        self.femur_r_df =  self.return_dataframe('femur_r')
        self.femur_l_df =  self.return_dataframe('femur_l')
        self.tibia_r_df =  self.return_dataframe('tibia_r')
        self.tibia_l_df =  self.return_dataframe('tibia_l')
        self.patella_r_df =  self.return_dataframe('patella_r')
        self.patella_l_df =  self.return_dataframe('patella_l')

    def import_stat_experim_markers(self):
        static_df = pd.read_csv(self.path_to_static, delimiter='\t', skiprows=3, header=[0,1], index_col=0)
        static_df=static_df.drop('Time', axis = 1)
            # adjust marker names
        old_lst = list(static_df.columns)
        new_lst = []
        for i, value in enumerate(old_lst):
            if old_lst[i][0][:2] != 'Un':
                remember = value[0]
                new_lst.append(value)
            else:
                new_lst.append((remember, old_lst[i][1]))
        static_df.columns = new_lst
            # calculate a series of mean values for each makrer's coordinate
        static_means = static_df.mean()[:-1]/1000
            # create a dataframe from experimental markers
        key = ""
        mean_dict = {}
        for i, entry in enumerate(static_means.index):
            if entry[0] != key:
                key = entry[0]
                mean_dict[key] = [static_means[i]]
            else:
                mean_dict[key].append(static_means[i])
        mean_skin_markers_df = (pd.DataFrame.from_dict(mean_dict)).T
        self.mean_skin_markers_df = mean_skin_markers_df.rename(columns={0:'x', 1:'y', 2:'z'})
            # lists of marker names
        self.torso_markers_names = ['C7', 'RBAK', 'CLAV', 'STRN', 'T10']
        self.pelvis_markers_names = ['RASI', 'RPSI', 'LPSI', 'LASI','PE01', 'PE02', 'PE03']
        self.femur_r_markers_names = ['RTH1', 'RTH2', 'RTH3', 'RKNE', 'RMKNE','RGT']
        self.femur_l_markers_names = ['LTH1', 'LTH2','LTH3', 'LKNE', 'LMKNE', 'LGT']
        self.tibia_r_markers_names = ['RTB1','RTB2', 'RTB3', 'RANK', 'RMMA']
        self.tibia_l_markers_names = ['LTB1', 'LTB2', 'LTB3', 'LANK', 'LMMA']
        self.foot_r_markers_names = ['RHEE', 'RTOE', 'RD5M']
        self.foot_l_markers_names = ['LHEE','LTOE', 'LD5M']
            # pd series of marker locations
        self.torso_static = self.mean_skin_markers_df.loc[self.torso_markers_names]
        self.pelvis_static = self.mean_skin_markers_df.loc[self.pelvis_markers_names]
        self.femur_r_static = self.mean_skin_markers_df.loc[self.femur_r_markers_names]
        self.femur_l_static  = self.mean_skin_markers_df.loc[self.femur_l_markers_names]
        self.tibia_r_static = self.mean_skin_markers_df.loc[self.tibia_r_markers_names]
        self.tibia_l_static = self.mean_skin_markers_df.loc[self.tibia_l_markers_names]
        self.foot_r_static = self.mean_skin_markers_df.loc[self.foot_r_markers_names]
        self.foot_l_static = self.mean_skin_markers_df.loc[self.foot_l_markers_names]

    def homologous_pairs(self):
            # lm pairs to match bone markers and experimental markers
        self.lm_pairs_for_transform = {'pelvis' : {'pelvis_df': ['ASIS_l', 'ASIS_r', 'PSIS_l', 'PSIS_r'], 'pelvis_static': ['LASI','RASI','LPSI','RPSI']},
                            
                            # pelvis_df points must be shifted by the appropriate femur center
                            'femur_r' : {'femur_r_df':['knee_r_lat','knee_r_med','gr_troch_lat_r'], 'pelvis_df': ['ASIS_r'],'femur_r_static':['RKNE', 'RMKNE','RGT'], 'pelvis_static': ['RASI']},
                            # pelvis_df points must be shifted by the appropriate femur center
                            'femur_l' : {'femur_l_df':['knee_l_lat','knee_l_med','gr_troch_lat_l'], 'pelvis_df': ['ASIS_l'],'femur_l_static':['LKNE', 'LMKNE','LGT'], 'pelvis_static': ['LASI']},
                            
                            # femur points femur_r_df must be shifted by the knee center
                            'tibia_r' : {'tibia_r_df':['ankle_r_lat', 'ankle_r_med',],'femur_r_df':['knee_r_lat','knee_r_med'], 'tibia_r_static':['RANK', 'RMMA'], 'femur_r_static':['RKNE', 'RMKNE']}, 
                            # femur points femur_l_df must be shifted by the knee center
                            'tibia_l' : {'tibia_l_df':['ankle_l_lat', 'ankle_l_med',],'femur_l_df':['knee_l_lat','knee_l_med'],'tibia_l_static':['LANK', 'LMMA'], 'femur_l_static':['LKNE', 'LMKNE']},
                            
                            # # foot and torso points are superimposed onto the projections in the OSIM scaled model
                            # 'foot_r' : {'imported_markers_df':['RHEE', 'RTOE', 'RD5M'],'foot_r_static':['RHEE', 'RTOE', 'RD5M']},
                            # 'foot_l' : {'imported_markers_df':['LHEE','LTOE', 'LD5M'],'foot_l_static':['LHEE','LTOE', 'LD5M']},
                            # 'torso' : {'imported_markers_df':['C7', 'RBAK', 'CLAV', 'STRN', 'T10'],'torso_static':['C7', 'RBAK', 'CLAV', 'STRN', 'T10']}
                            }

    def fit_pelvis(self):
            # create transform for pelvis
        from_ = self.pelvis_static.loc[self.lm_pairs_for_transform['pelvis']['pelvis_static']].to_numpy()
        to_ = self.pelvis_df.loc[self.lm_pairs_for_transform['pelvis']['pelvis_df']].to_numpy()
        rotation, translation1, translation2 = self.transforms(to_, from_)
            # transform pelvis markers
        translated = self.pelvis_static.to_numpy() - translation1
        rotated = np.matmul(rotation, translated.T).T
        final = rotated + translation2
            # record results
        self.pelvis_new_markers_df = pd.DataFrame(final, index=self.pelvis_markers_names, columns=['x', 'y', 'z'])


    def fit_femur_r(self):
            # create transform for femur_r
        from_ = (pd.concat([self.femur_r_static.loc[self.lm_pairs_for_transform['femur_r']['femur_r_static']], 
                            self.pelvis_static.loc[self.lm_pairs_for_transform['femur_r']['pelvis_static']]
                            ])).to_numpy()
        to_ = (pd.concat([self.femur_r_df.loc[self.lm_pairs_for_transform['femur_r']['femur_r_df']], 
                        (self.pelvis_df.loc[self.lm_pairs_for_transform['femur_r']['pelvis_df']] - self.pelvis_df.loc['femur_r_center_in_pelvis'])
                        ])).to_numpy()
        rotation, translation1, translation2 = self.transforms(to_, from_)       
            # transform femur_r markers
        translated = self.femur_r_static.to_numpy() - translation1
        rotated = np.matmul(rotation, translated.T).T
        final = rotated + translation2
            # record results
        self.femur_r_new_markers_df = pd.DataFrame(final, index=self.femur_r_markers_names, columns=['x', 'y', 'z'])

    def fit_femur_l(self):
            # create transform for femur_r
        from_ = (pd.concat([self.femur_l_static.loc[self.lm_pairs_for_transform['femur_l']['femur_l_static']], 
                            self.pelvis_static.loc[self.lm_pairs_for_transform['femur_l']['pelvis_static']]
                            ])).to_numpy()
        to_ = (pd.concat([self.femur_l_df.loc[self.lm_pairs_for_transform['femur_l']['femur_l_df']], 
                        (self.pelvis_df.loc[self.lm_pairs_for_transform['femur_l']['pelvis_df']] - self.pelvis_df.loc['femur_l_center_in_pelvis'])
                        ])).to_numpy()
        rotation, translation1, translation2 = self.transforms(to_, from_)
            # transform femur_l markers
        translated = self.femur_l_static.to_numpy() - translation1
        rotated = np.matmul(rotation, translated.T).T
        final = rotated + translation2
            # record results
        self.femur_l_new_markers_df = pd.DataFrame(final, index=self.femur_l_markers_names, columns=['x', 'y', 'z'])

    def fit_tibia_r(self):
            # assemble data for bone points
        knee_r = self.femur_r_df.loc[self.lm_pairs_for_transform['tibia_r']['femur_r_df'], ['x','y','z']] - self.femur_r_df.loc['knee_r_center_in_femur_r']
        ankle_r = self.tibia_r_df.loc[self.lm_pairs_for_transform['tibia_r']['tibia_r_df'], ['x','y','z']]
        shin_r = pd.concat([knee_r, ankle_r])
        to_ = shin_r.to_numpy()
        to_ = np.array(to_, dtype=np.float64)
            # create according data for skin
        skin_markers = self.mean_skin_markers_df.loc[self.lm_pairs_for_transform['tibia_r']['femur_r_static'] + self.lm_pairs_for_transform['tibia_r']['tibia_r_static']]
        from_ = skin_markers.to_numpy()
            # create transform
        rotation, translation1, translation2 = self.transforms(to_, from_)
            # transform tibia_r markers
        translated = self.tibia_r_static.to_numpy() - translation1
        rotated = np.matmul(rotation, translated.T).T
        final = rotated + translation2
            # record results
        self.tibia_r_new_markers_df = pd.DataFrame(final, index=self.tibia_r_markers_names, columns=['x', 'y', 'z'])
        print('tibia_r is done')

    def fit_tibia_l(self):
            # assemble data for bone points
        knee_l = self.femur_l_df.loc[self.lm_pairs_for_transform['tibia_l']['femur_l_df'], ['x','y','z']] - self.femur_l_df.loc['knee_l_center_in_femur_l']
        ankle_l = self.tibia_l_df.loc[self.lm_pairs_for_transform['tibia_l']['tibia_l_df'], ['x','y','z']]
        shin_l = pd.concat([knee_l, ankle_l])
        to_ = shin_l.to_numpy()
        to_ = np.array(to_, dtype=np.float64)
            # create according data for skin
        skin_markers = self.mean_skin_markers_df.loc[self.lm_pairs_for_transform['tibia_l']['femur_l_static'] + self.lm_pairs_for_transform['tibia_l']['tibia_l_static']]
        from_ = skin_markers.to_numpy()
            # create transform
        rotation, translation1, translation2 = self.transforms(to_, from_)
            # transform tibia_r markers
        translated = self.tibia_l_static.to_numpy() - translation1
        rotated = np.matmul(rotation, translated.T).T
        final = rotated + translation2
            # record results
        self.tibia_l_new_markers_df = pd.DataFrame(final, index=self.tibia_l_markers_names, columns=['x', 'y', 'z'])
        print('tibia_l is done')
    # def fit_torso(self):
    #         # create transform for torso
    #     from_ = np.array(self.torso_static.loc[self.lm_pairs_for_transform['torso']['torso_static']].to_numpy() , dtype=np.float64)
    #     to_ = np.array(self.imported_markers_df.loc[self.lm_pairs_for_transform['torso']['imported_markers_df'], ['x', 'y', 'z']].to_numpy(), dtype=np.float64)
    #     rotation, translation1, translation2 = self.transforms(to_, from_)
    #         # transform torso markers
    #     translated = self.torso_static.to_numpy() - translation1
    #     rotated = np.matmul(rotation, translated.T).T
    #     final = rotated + translation2
    #         # record results
    #     self.torso_new_markers_df = pd.DataFrame(final, index=self.torso_markers_names, columns=['x', 'y', 'z'])


    def update_model(self):
            # execute transformations
        self.fit_pelvis()
        self.fit_femur_l()
        self.fit_femur_r()
        self.fit_tibia_l()
        self.fit_tibia_r()
        # self.fit_torso()
            # concatenate all new dataframes
        new_skin_mrkrs_df = pd.concat([self.pelvis_new_markers_df, self.femur_r_new_markers_df, self.femur_l_new_markers_df, self.tibia_r_new_markers_df, self.tibia_l_new_markers_df]) # self.torso_new_markers_df, 
            # parse model to update
        tree=ET.parse(self.model_to_update)
        root = tree.getroot()
            # update markers
        for marker in root.iter('Marker'):
            name = marker.attrib['name']
            if name in new_skin_mrkrs_df.index:
                new_text = f"{new_skin_mrkrs_df.loc[name, 'x']} {new_skin_mrkrs_df.loc[name, 'y']} {new_skin_mrkrs_df.loc[name, 'z']}"
                marker.find('location').text = new_text
        # record updated model
        tree.write(self.updated_model)

        print('tree command is run',self.updated_model)