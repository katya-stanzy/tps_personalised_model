import opensim as osim
import numpy as np
from scipy import linalg, optimize
from sklearn.metrics import mean_squared_error
from time import time
import sys
import re
import os
from itertools import product
import logging
from pathlib import Path

import xml.etree.ElementTree as ET

# sys.path.append('/opt/opensim-core/lib/python3.6/site-packages/')


# get model joint definitions
def getModelJointDefinitions(model_file):
    tree_model = ET.parse(model_file)
    root_model = tree_model.getroot()
    jointStructure={}
    for jointSet in root_model.iter('JointSet'):
        objects = jointSet.iter('objects')
        for obj in objects:
            numJoints = len(obj)
        for i in range(numJoints):
            name = obj[i].attrib['name']
            parentFrame = obj[i].find('socket_parent_frame').text
            childFrame = obj[i].find('socket_child_frame').text
            parentBody = parentFrame[:-len('_offset')]
            childBody = childFrame[:-len('_offset')]
            jointStructure[name] = {'type': obj[i].tag, 'parentFrame':parentFrame, 'childFrame':childFrame, 'parentBody':parentBody, 'childBody':childBody}
    return jointStructure

# get child body joint
# loop through joints to check the child body entry and return if true
def getChildBodyJoint(jointStructure, childBody):
    allJoints = jointStructure.keys()
    for joint in allJoints:      
        if jointStructure[joint]['childBody'] == childBody:
            return joint
        
# get parent body joint
# loop through joints to check the parent body entry and return if true
def getParentBodyJoint(jointStructure, parentBody):
    allJoints = jointStructure.keys()
    for joint in allJoints:      
        if jointStructure[joint]['parentBody'] == parentBody:
            return joint
        
# get muscle attach body
def getMuscleAttachBody(osimModel, musclePathPointName):
    # Functon to return the name of the muscel path point from a specified model
    # and muscle, where the specified body is the parent body
    
    # Written by Emiliano Ravera emiliano.ravera@uner.edu.ar as part of the 
    # Python version of work by Luca Modenese in the parameterisation of muscle
    # tendon properties.
    
    # Input: OpenSim model objects
    # Output: bodyName - the body for the specified musclepath
    
    bodyName = []
    # load model XML file  
    osimModel_filepath = osimModel.getInputFileName()
    osimModel_file = ET.parse(osimModel_filepath)
    model = osimModel_file.getroot()
    
    musclePath = model.findall('./' + musclePathPointName)
    
    for musclePath in model.findall('.//PathPoint'):
        if musclePath.get('name') == musclePathPointName:
            bodyName = re.sub('/bodyset/', '', musclePath.find('socket_parent_frame').text)
            
    for musclePath in model.findall('.//ConditionalPathPoint'):
        if musclePath.get('name') == musclePathPointName:
            bodyName = re.sub('/bodyset/', '', musclePath.find('socket_parent_frame').text)
    
    for musclePath in model.findall('.//MovingPathPoint'):
        if musclePath.get('name') == musclePathPointName:
            bodyName = re.sub('/bodyset/', '', musclePath.find('socket_parent_frame').text)
        
    return bodyName

# get joint spanned by muscle
def getJointsSpannedByMuscle(model_file_name, OSMuscleName):
    # Given as INPUT a muscle OSMuscleName from an OpenSim model, this function
    # returns the OUTPUT list jointNameSet containing the OpenSim jointNames
    # crossed by the OSMuscle.
    # 
    # It works through the following steps:
    #   1) extracts the GeometryPath
    #   2) loops through the single points, determining the body they belong to
    #   3) stores the bodies to which the muscle points are attached to
    #   4) determines the nr of joints based on body indexes
    #   5) stores the crossed OpenSim joints in the output list named jointNameSet
    #
    # NB this function return the crossed joints independently on the
    # constraints applied to the coordinates. Eg patello-femoral is considered as a
    # joint, although in Arnold's model it does not have independent
    # coordinates, but it is moved in dependency of the knee flexion angle.
    
    # Written by Emiliano Ravera emiliano.ravera@uner.edu.ar as part of the 
    # Python version of work by Luca Modenese in the parameterisation of muscle
    # tendon properties.

    osimModel = osim.Model(model_file_name)
    
    # useful initializations
    BodySet = osimModel.getBodySet()
    muscle  = osimModel.getMuscles().get(OSMuscleName)
    
    # additions BAK -> adapted by ER
    # load a jointStrucute detailing bone and joint configurations
    # osimModel_file = osimModel.getInputFileName()
    jointStructure = getModelJointDefinitions(model_file_name)
    
    # Extracting the PathPointSet via GeometryPath
    musclePath = muscle.getGeometryPath()
    musclePathPointSet = musclePath.getPathPointSet()
    
    # for loops to get the attachment bodies
    muscleAttachBodies = []
    muscleAttachIndex = []
    
    for n_point in range(0, musclePathPointSet.getSize()):
        
        # get the current muscle point
        muscelPathPoint_name =  musclePathPointSet.get(n_point).getName()
        currentAttachBody = getMuscleAttachBody(osimModel, muscelPathPoint_name)
        
        # Initialize
        if n_point == 0:
            previousAttachBody = currentAttachBody
            muscleAttachBodies.append(currentAttachBody)
            muscleAttachIndex.append(BodySet.getIndex(currentAttachBody))
        # building a list of the bodies attached to the muscles
        if currentAttachBody != previousAttachBody:
            muscleAttachBodies.append(currentAttachBody)
            muscleAttachIndex.append(BodySet.getIndex(currentAttachBody))
            previousAttachBody = currentAttachBody
            
    # end of loops to get the attacement bodies
            
    # From distal body checking the joint names going up until the desired
    # OSJointName is found or the proximal body is reached as parent body.
    DistalBodyName = muscleAttachBodies[-1]
    bodyName = DistalBodyName
    ProximalBodyName = muscleAttachBodies[0]
    body =  BodySet.get(DistalBodyName)
    
    spannedJointNameOld = ''
    NoDofjointNameSet = []
    jointNameSet = []
    
    while bodyName != ProximalBodyName:
            
        # BAK implementation -> adapted by ER
        spannedJointName = getChildBodyJoint(jointStructure, body.getName())
        spannedJoint = osimModel.getJointSet().get(spannedJointName)
        
        if spannedJointName == spannedJointNameOld:
            # BAK implementation -> adapted by ER
            body = osimModel.getBodySet().get(bodyName)
            spannedJointNameOld = spannedJointName
        else:
            if spannedJoint.numCoordinates() != 0:
                jointNameSet.append(spannedJointName)
            else:
                NoDofjointNameSet.append(spannedJointName)
            
            spannedJointNameOld = spannedJointName
            bodyName = jointStructure[spannedJointName]['parentBody']
            body = osimModel.getBodySet().get(bodyName)
            
        bodyName = body.getName()
    
     
    if not jointNameSet:
        print('ERORR: ' + 'No joint detected for muscle ' + OSMuscleName)
    
    if NoDofjointNameSet:
        for value in NoDofjointNameSet:
            print('Joint ' + value + ' has no dof.')
    
    varargout = NoDofjointNameSet 
    
    
    return jointNameSet, varargout

# get indep coord and joint
def getIndepCoordAndJoint(osimModel, constraint_coord_name):
    # Function that given a dependent coordinate finds the independent
    # coordinate and the associated joint. The function assumes that the
    # constraint is a CoordinateCoupleConstraint as used by Arnold, Delp and
    # LLLM. The function can be useful to manage the patellar joint for instance.
    
    # Input: OpenSim model objects
    # Output: ind_coord_name and ind_coord_joint_name - the joint with specific constraint
    
    # Written by Emiliano Ravera emiliano.ravera@uner.edu.ar as part of the 
    # Python version of work by Luca Modenese in the parameterisation of muscle
    # tendon properties.
    
    ind_coord_name = ''
    ind_coord_joint_name = ''
    
    # load model XML file  
    osimModel_filepath = osimModel.getInputFileName()
    osimModel_file = ET.parse(osimModel_filepath)
    model = osimModel_file.getroot()
        
    # double check: if not constrained then function returns
    flag = [1  for constraint in  model.findall('.//CoordinateCouplerConstraint') if constraint.get('name').find(constraint_coord_name)]
       
    if flag == []:
        # print(constraint_coord_name + ' is not a constrained coordinate.')
        logging.error(' ' + constraint_coord_name + ' is not a constrained coordinate.')
        return ind_coord_name, ind_coord_joint_name
    
    # otherwise search through the constraints
    for constraint in  model.findall('.//CoordinateCouplerConstraint'):
        
        # this function assumes that the constraint will be a coordinate
        # coupler contraint ( Arnold's model and LLLM uses this)
                
        # get dep coordinate and check if it is the coord of interest
        dep_coord_name = constraint.find('dependent_coordinate_name').text
        
        if dep_coord_name in constraint_coord_name:
            # print('WARNING: Only one indipendent coordinate is managed by the "getIndepCoordAndJoint" function yet.')
            logging.warning(' Only one indipendent coordinate is managed by the "getIndepCoordAndJoint" function yet.')
            
            ind_coord_name = constraint.find('independent_coordinate_names').text
            ind_coord_joint_name = constraint.find('independent_coordinate_names').text # assume the same name for coordinate and joint 
            
    return ind_coord_name, ind_coord_joint_name

# sample muscle quantities
def sampleMuscleQuantities(osimModel,OSMuscle,muscleQuant, N_EvalPoints):
    # Given as INPUT an OpenSim muscle OSModel, OSMuscle, a muscle variable and a nr
    # of evaluation points this function returns as
    # musOutput a vector of the muscle variable of interest
    # obtained by sampling the ROM of the joint spanned by the muscle in
    # N_EvalPoints evaluation points.
    # For multidof joint the combinations of ROMs are considered.
    # For multiarticular muscles the combination of ROM are considered.
    # The script is totally general because based on generating strings of code 
    # correspondent to the encountered code. The strings are evaluated at the end.
    
    # IMPORTANT 1
    # The function can decrease the N_EvalPoints if there are too many dof 
    # involved (ASSUMPTION is that a better sampling will be vanified by the 
    # huge amount of data generated). This is an option that can be controlled
    # by the user by deciding if to use it or not (setting limit_discr = 0/1)
    # and, in case the sampling is limited, by setting a lower limit to the
    # discretization.
    
    # IMPORTANT 2
    # Another check is done on the dofs: only INDEPENDENT coordinate are
    # considered. This is fundamental for patellofemoral joint that both in
    # LLLM and Arnold's model are constrained dof, dependent on the knee
    # flexion angle. This function assumes to be used with Arnold's model.
    # currentState is initialState;
    
    # IMPORTANT 3
    # At the purposes of the muscle optimizer it is important that here the
    # model is initialize every time. So there are no risk of working with an
    # old state (important for Schutte muscles for instance, where we observed
    # that it was necessary to re-initialize the muscle after updating the
    # muscle parameters).
    
    # Written by Emiliano Ravera emiliano.ravera@uner.edu.ar as part of the 
    # Python version of work by Luca Modenese in the parameterisation of muscle
    # tendon properties.
    
    # ======= SETTINGS ======
    # limit (1) or not (0) the discretization of the joint space sampling
    limit_discr = 0
    # minimum angular discretization
    min_increm_in_deg = 1
    # =======================
    
    # initialize the model
    currentState = osimModel.initSystem()
    
    # getting the joint crossed by a muscle
    muscleCrossedJointSet, _ = getJointsSpannedByMuscle(osimModel.getInputFileName(), OSMuscle.getName())
            
    # index for effective dofs
    DOF_Index = []
    CoordinateBoundaries = []
    degIncrem = []
    
    for _, curr_joint in enumerate(muscleCrossedJointSet):
        # Initial estimation of the nr of Dof of the CoordinateSet for that
        # joint before checking for locked and constraint dofs.
        nDOF = osimModel.getJointSet().get(curr_joint).numCoordinates()
        
        # skip welded joint and removes welded joint from muscleCrossedJointSet
        if nDOF == 0:
            continue
        
        # calculating effective dof for that joint
        effect_DOF = nDOF
        for n_coord in range(0,nDOF):
            # get coordinate
            curr_coord = osimModel.getJointSet().get(curr_joint).get_coordinates(n_coord)
            curr_coord_name = curr_coord.getName()
            
            # skip dof if locked
            if curr_coord.getLocked(currentState):
                continue
            
            # if coordinate is constrained then the independent coordinate and
            # associated joint will be listed in the sampling "map"
            if curr_coord.isConstrained(currentState) and not curr_coord.getLocked(currentState):
                constraint_coord_name = curr_coord_name
                # finding the independent coordinate
                ind_coord_name, ind_coord_joint_name = getIndepCoordAndJoint(osimModel, constraint_coord_name)
                # updating the coordinate name to be saved in the list
                curr_coord_name = ind_coord_name
                effect_DOF -= 1
                # ignoring constrained dof if they point to an independent
                # coordinate that has already been stored
                if osimModel.getCoordinateSet().getIndex(curr_coord_name) in DOF_Index:
                    continue
                # skip dof if independent coordinate locked (the coord
                # correspondent to the name needs to be extracted)
                if osimModel.getCoordinateSet().get(curr_coord_name).getLocked(currentState):
                    continue
                
            # NB: DOF_Index is used later in the string generated code.
            # CRUCIAL: the index of dof now is model based ("global") and
            # different from the joint based used until now.
            DOF_Index.append(osimModel.getCoordinateSet().getIndex(curr_coord_name))
            
            # necessary update/reload the curr_coord to avoid problems with 
            # dependent coordinates
            curr_coord = osimModel.getCoordinateSet().get(DOF_Index[-1])
            
            # Getting the values defining the range
            jointRange = np.zeros(2)
            jointRange[0] = curr_coord.getRangeMin()
            jointRange[1] = curr_coord.getRangeMax()
            
            # Storing range of motion conveniently
            CoordinateBoundaries.append(jointRange)
            
            # increments in the variables when sampling the mtl space. 
            # Increments are different for each dof and based on N_eval.
            # Defining the increments
            degIncrem.append((jointRange[1] - jointRange[0]) / (N_EvalPoints-1))
            
            # limit or not the discretization of the joint space sampling
            # a limit to the increase can be set though
            if limit_discr == 1 and degIncrem[-1] < np.radians(min_increm_in_deg):
                degIncrem[-1] = np.radians(min_increm_in_deg)
    
        
    # assigns an interval of variation following the initial and final value
    # for each dof X
        
    # setting up for loops in order to explore all the possible combination of
    # joint angles (looping on all the dofs of each joint for all the joint
    # crossed by the muscle).
    # The model pose is updated via: " coordToUpd.setValue(currentState,setAngleDof)"
    # The right dof to update is chosen via: "coordToUpd = osimModel.getCoordinateSet.get(n_instr)"
    
    # generate a dictionary with CoordinateRange for each dof X. 
    # The dictionary keys are the DOF_Index in the model
    CoordinateRange = {}
    for pos, dof in enumerate(DOF_Index):
        CoordinateRange[str(dof)] = np.linspace(CoordinateBoundaries[pos][0] , CoordinateBoundaries[pos][1], N_EvalPoints)
    
    # generate a list of dictionaries to explore all the possible combination of
    # joit angle
    CoordinateCombinations = [dict(zip(CoordinateRange.keys(), element)) for element in product(*CoordinateRange.values())] 
    
    # looping on all the dofs combinations
    musOutput = [None] * len(CoordinateCombinations)
    
    for iteration, DOF_comb in enumerate(CoordinateCombinations):
        # Set the model pose
        for dof_ind in DOF_comb.keys():
            coordToUpd = osimModel.getCoordinateSet().get(int(dof_ind))
            coordToUpd.setValue(currentState, CoordinateCombinations[iteration][dof_ind])
        
        # calculating muscle length for the muscle    
        if muscleQuant == 'MTL':
            musOutput[iteration] = OSMuscle.getGeometryPath().getLength(currentState)
            
        if muscleQuant == 'LfibNorm':
            OSMuscle.setActivation(currentState,1.0)
            osimModel.equilibrateMuscles(currentState)
            musOutput[iteration] = OSMuscle.getNormalizedFiberLength(currentState)
            
        if muscleQuant == 'Lten':
            OSMuscle.setActivation(currentState,1.0)
            osimModel.equilibrateMuscles(currentState)
            musOutput[iteration] = OSMuscle.getTendonLength(currentState)
            
        if muscleQuant == 'Ffib':
            OSMuscle.setActivation(currentState,1.0)
            osimModel.equilibrateMuscles(currentState)
            musOutput[iteration] = OSMuscle.getActiveFiberForce(currentState)
            
        if muscleQuant == 'all':
            OSMuscle.setActivation(currentState,1.0)
            osimModel.equilibrateMuscles(currentState)
            musOutput[iteration] = [ OSMuscle.getGeometryPath().getLength(currentState), \
                                    OSMuscle.getNormalizedFiberLength(currentState), \
                                    OSMuscle.getTendonLength(currentState), \
                                    OSMuscle.getActiveFiberForce(currentState), \
                                    OSMuscle.getPennationAngle(currentState) ]

    return musOutput

# optimise muscle parameters
# This function optimizes the muscle parameters as described in Modenese L, 
# Ceseracciu E, Reggiani M, Lloyd DG (2015). Estimation of 
# musculotendon parameters for scaled and subject specific musculoskeletal 
# models using an optimization technique. Journal of Biomechanics (submitted)
# and prints the results to command window.
# Also it stores information about the optimization in the structure SimInfo

# Written by Emiliano Ravera emiliano.ravera@uner.edu.ar as part of the 
# Python version of work by Luca Modenese in the parameterisation of muscle
# tendon properties.

#-------------------------------- 

def optimMuscleParams(osimModel_ref_filepath, osimModel_targ_filepath, N_eval, log_folder):
    
    
    # results file identifier
    res_file_id_exp = '_N' + str(N_eval)
    
    # import models
    osimModel_ref = osim.Model(osimModel_ref_filepath)
    osimModel_targ = osim.Model(osimModel_targ_filepath)
    
    # models details
    name = Path(osimModel_targ_filepath).stem
    ext = Path(osimModel_targ_filepath).suffix
    
    # assigning new name to the model
    osimModel_opt_name = name + '_opt' + res_file_id_exp + ext
    osimModel_targ.setName(osimModel_opt_name)
    
    # initializing log file
    #log_folder = 'logging'
    working_dir = os.getcwd()
    path = os.path.join(working_dir, log_folder)
    if not os.path.exists(path):
        os.makedirs(path)
    log_file = os.path.join(log_folder, 'log_file.txt')
    if not os.path.exists(log_file):
        f = open(log_file, 'w')
        f.close()

    def logging_message(message):
        with open(log_file, 'a') as f:
            f.write(message)


    # logger = logging.getLogger(__name__)
    # #logging.basicConfig(filename = str(path) + '/' + name + '_opt' + res_file_id_exp +'.log', filemode = 'w', format = '%(levelname)s:%(message)s', level = logging.INFO)
    # logging.basicConfig(filename = str(path) + '/' + 'file.log', encoding='utf-8', level=logging.INFO)
    # print(str(path) + '/' + 'file.log')
    
    # get muscles
    muscles = osimModel_ref.getMuscles()
    muscles_scaled = osimModel_targ.getMuscles()
    
    # initialize with recognizable values
    LmOptLts_opt = -1000*np.ones((muscles.getSize(),2))
    SimInfo = {}
    
    for n_mus in range(0, muscles.getSize()):
        
        tic = time()
        
        # current muscle name (here so that it is possible to choose a single muscle when developing).
        curr_mus_name = muscles.get(n_mus).getName()
        print('processing mus ' + str(n_mus+1) + ': ' + curr_mus_name)
        
        # import muscles
        curr_mus = muscles.get(curr_mus_name)
        curr_mus_scaled = muscles_scaled.get(curr_mus_name)
        
        # extracting the muscle parameters from reference model
        LmOptLts = [curr_mus.getOptimalFiberLength(), curr_mus.getTendonSlackLength()]
        PenAngleOpt = curr_mus.getPennationAngleAtOptimalFiberLength()
        Mus_ref = sampleMuscleQuantities(osimModel_ref,curr_mus,'all',N_eval)
        
        # calculating minimum fiber length before having pennation 90 deg
        # acos(0.1) = 1.47 red = 84 degrees, chosen as in OpenSim
        limitPenAngle = np.arccos(0.1)
        # this is the minimum length the fiber can be for geometrical reasons.
        LfibNorm_min = np.sin(PenAngleOpt) / np.sin(limitPenAngle)
        # LfibNorm as calculated above can be shorter than the minimum length
        # at which the fiber can generate force (taken to be 0.5 Zajac 1989)
        if LfibNorm_min < 0.5:
           LfibNorm_min = 0.5
        
        # muscle-tendon paramenters value
        MTL_ref = [musc_param_iter[0] for musc_param_iter in Mus_ref]
        #print('MTL_ref', MTL_ref)
        LfibNorm_ref = [musc_param_iter[1] for musc_param_iter in Mus_ref]
        #print('LfibNorm_ref', LfibNorm_ref)
        LtenNorm_ref = [musc_param_iter[2]/LmOptLts[1] for musc_param_iter in Mus_ref]
        #print('LtenNorm_ref',LtenNorm_ref)
        penAngle_ref = [musc_param_iter[4] for musc_param_iter in Mus_ref]
        #print('penAngle_ref',penAngle_ref)
        # LfibNomrOnTen_ref = LfibNorm_ref.*cos(penAngle_ref)
        LfibNomrOnTen_ref = [(musc_param_iter[1]*np.cos(musc_param_iter[4])) for musc_param_iter in Mus_ref]  
        #print('LfibNomrOnTen_ref',LfibNomrOnTen_ref)       
        
        # checking the muscle configuration that do not respect the condition.
        okList = [pos for pos, value in enumerate(LfibNorm_ref) if value > LfibNorm_min]
        print('okList', len(okList))
        # keeping only acceptable values
        MTL_ref = np.array([MTL_ref[index] for index in okList])
        LfibNorm_ref = np.array([LfibNorm_ref[index] for index in okList])
        LtenNorm_ref = np.array([LtenNorm_ref[index] for index in okList])
        penAngle_ref = np.array([penAngle_ref[index] for index in okList])
        LfibNomrOnTen_ref = np.array([LfibNomrOnTen_ref[index] for index in okList])
        
        # in the target only MTL is needed for all muscles
        MTL_targ = sampleMuscleQuantities(osimModel_targ,curr_mus_scaled,'MTL',N_eval)
        evalTotPoints = len(MTL_targ)
        MTL_targ = np.array([MTL_targ[index] for index in okList])
        evalOkPoints  = len(MTL_targ)
        
        # The problem to be solved is: 
        # [LmNorm*cos(penAngle) LtNorm]*[Lmopt Lts]' = MTL;
        # written as Ax = b or their equivalent (A^T A) x = (A^T b)  
        A = np.array([LfibNomrOnTen_ref , LtenNorm_ref]).T
        b = MTL_targ
        
        # ===== LINSOL =======
        # solving the problem to calculate the muscle param 
        x = linalg.solve(np.dot(A.T , A) , np.dot(A.T , b))
        LmOptLts_opt[n_mus] = x
        
        # checking the results
        if np.min(x) <= 0:
            # informing the user
            line0 = ' '
            line1 = 'Negative value estimated for muscle parameter of muscle ' + curr_mus_name + '\n'
            line2 = '                         Lm Opt        Lts' + '\n'
            line3 = 'Template model       : ' + str(LmOptLts) + '\n'
            line4 = 'Optimized param      : ' + str(LmOptLts_opt[n_mus]) + '\n'
            line5 = 'Template model muscle/tendon: ' + str(LmOptLts[0]/LmOptLts[1]) + '\n'
            line6 = 'Optimized param muscle/tendon: ' + str(LmOptLts_opt[n_mus][0]/LmOptLts_opt[n_mus][1]) + '\n'

            # ===== IMPLEMENTING CORRECTIONS IF ESTIMATION IS NOT CORRECT =======
            x = optimize.nnls(np.dot(A.T , A) , np.dot(A.T , b))
            x = x[0]
            LmOptLts_opt[n_mus] = x
            line7 = 'Opt params (optimize.nnls): ' + str(LmOptLts_opt[n_mus])
            
            logging_message(line0 + line1 + line2 + line3 + line4 + line7 + '\n') #+ line5 + line6 

            #logger.info(line0 + line1 + line2 + line3 + line4 + line5 + '\n')
            # In our tests, if something goes wrong is generally tendon slack 
            # length becoming negative or zero because tendon length doesn't change
            # throughout the range of motion, so lowering the rank of A.

            if np.min(x) <= 0:
                # analyzes of Lten behaviour
                Lten_ref = [musc_param_iter[2] for musc_param_iter in Mus_ref]
                Lten_ref = np.array([Lten_ref[index] for index in okList])
                if (np.max(Lten_ref) - np.min(Lten_ref)) < 0.0001:
                    logging_message(' Tendon length not changing throughout range of motion')
                    #logger.warning(' Tendon length not changing throughout range of motion')
                
                # calculating proportion of tendon and fiber
                Lten_fraction = Lten_ref/MTL_ref
                Lten_targ = Lten_fraction*MTL_targ
                print('Lten_targ', Lten_targ)
                
                # first round: optimizing Lopt maintaing the proportion of
                # tendon as in the reference model
                A1 = np.array([LfibNomrOnTen_ref , LtenNorm_ref*0]).T
                b1 = MTL_targ - Lten_targ
                x1 = optimize.nnls(np.dot(A1.T , A1) , np.dot(A1.T , b1))
                x[0] = x1[0][0]
                
                # second round: using the optimized Lopt to recalculate Lts
                A2 = np.array([LfibNomrOnTen_ref*0 , LtenNorm_ref]).T
                b2 = MTL_targ - np.dot(A1,x1[0])
                x2 = optimize.nnls(np.dot(A2.T , A2) , np.dot(A2.T , b2))
                x[1] = x2[0][1]
                
                LmOptLts_opt[n_mus] = x
                print('LmOptLts_opt[n_mus]', LmOptLts_opt[n_mus])
            
        
        # Here tests about/against optimizers were implemented
        
        # calculating the error (mean squared errors)
        fval = mean_squared_error(b, np.dot(A,x), squared=False)
        
        # update muscles from scaled model
        #print('old_muscle',curr_mus_scaled.getOptimalFiberLength(), 'old_tendon', curr_mus_scaled.getTendonSlackLength())
        #print('old_muscle/tendon',curr_mus_scaled.getOptimalFiberLength()/curr_mus_scaled.getTendonSlackLength())

        curr_mus_scaled.setOptimalFiberLength(LmOptLts_opt[n_mus][0])
        curr_mus_scaled.setTendonSlackLength(LmOptLts_opt[n_mus][1])
        #print('new_muscle', LmOptLts_opt[n_mus][0], 'new_tendon', LmOptLts_opt[n_mus][1])
        #print('new_muscle/tendon', LmOptLts_opt[n_mus][0]/LmOptLts_opt[n_mus][1])
        
        # PRINT LOGS
        toc = time() - tic
        line0 = ' '
        line1 = 'Calculated optimized muscle parameters for ' + curr_mus.getName() + ' in ' +  str(toc) + ' seconds.' + '\n'
        line2 = '                         Lm Opt        Lts' + '\n'
        line3 = 'Template model       : ' + str(LmOptLts) + '\n'
        line4 = 'Optimized param      : ' + str(LmOptLts_opt[n_mus]) + '\n'
        line5 = 'Nr of eval points    : ' + str(evalOkPoints) + '/' + str(evalTotPoints) + ' used' + '\n'
        line6 = 'fval                 : ' + str(fval) + '\n'
        line7 = 'var from template [%]: ' + str(100*(np.abs(LmOptLts - LmOptLts_opt[n_mus])) / LmOptLts) + '%' + '\n'      
        line8 = 'Template model muscle/tendon: ' + str(LmOptLts[0]/LmOptLts[1]) + '\n'
        line9 = 'Optimized param muscle/tendon: ' + str(LmOptLts_opt[n_mus][0]/LmOptLts_opt[n_mus][1]) + '\n'

        logging_message(line0 + line1 + line2 + line3 + line4 + line5 + line6 + line7 + '\n') # + line8 + line9
        #logger.info(line0 + line1 + line2 + line3 + line4 + line5 + line6 + line7 + '\n')
              
        # SIMULATION INFO AND RESULTS
        
        SimInfo[n_mus] = {}
        SimInfo[n_mus]['colheader'] = curr_mus.getName()
        SimInfo[n_mus]['LmOptLts_ref'] = LmOptLts
        SimInfo[n_mus]['LmOptLts_opt'] = LmOptLts_opt[n_mus]
        SimInfo[n_mus]['varPercLmOptLts'] = 100*(np.abs(LmOptLts - LmOptLts_opt[n_mus])) / LmOptLts
        SimInfo[n_mus]['sampledEvalPoints'] = evalOkPoints
        SimInfo[n_mus]['sampledEvalPoints'] = evalTotPoints
        SimInfo[n_mus]['fval'] = fval
        
    # assigning optimized model as output
    osimModel_opt = osimModel_targ
            
    return osimModel_opt, SimInfo