'''
Structure builder for cove- and gulf-edges zigzag graphene nanoribbons
Author: Florian Arnold
Affiliation: TU Dresden, Germany
'''

#========#
# Import #
#========#

import argparse               # parser for command line
import numpy as np            # mathematical operations
import sys                    # system operations
import random                 # RNG
from scipy import spatial     # used to get neighbor lists
import copy                   # deepcopy of lists
import os                     # file system operations


#===========#
# Constants #
#===========#

### structural properties
DIST_CC = 1.4 # Å, C-C distance
DIST_CH = 1.1 # Å, C-H distance
VACUUM = 20 # Å, vacuum gap between C atoms
OOP_MAX = 0.05 # Å, maximum displacement oop

### auxiliary variables
SIN30, COS30 = np.sin(30/360*2*np.pi), np.cos(30/360*2*np.pi) # sin and cos of 30°
SIN60, COS60 = np.sin(60/360*2*np.pi), np.cos(60/360*2*np.pi) # sin and cos of 60°
SIN90, COS90 = np.sin(90/360*2*np.pi), np.cos(90/360*2*np.pi) # sin and cos of 90°
SIN120, COS120 = np.sin(120/360*2*np.pi), np.cos(120/360*2*np.pi) # sin and cos of 60°


#========================#
# Command line interface #
#========================#

### create parser object
parser = argparse.ArgumentParser(description='Create ZGNR-C/G structure.') # https://docs.python.org/3.3/library/argparse.html

### add options to parser
## option 1: ribbon width
parser.add_argument("-N", nargs=1, type=int, help="Width of N-ZGNR backbone", metavar="[N]")
## option 2: parameters for ZGNR-C (cove-type)
parser.add_argument("-C", nargs=2, type=float, help="Parameters for N-ZGNR-C(a,b)", metavar=("[a]", "[b]"))
## option 3: parameters for ZGNR-G (gulf-type)
parser.add_argument("-G", nargs=3, type=float, help="Parameters for N-ZGNR-G_M(a,b)", metavar=("[a]", "[b]", "[M]"))
## option 4: number of repetition units of ZGNR-C/G (bulk in heterojunctions)
parser.add_argument("-R", nargs=1, type=int, help="Number primitive unit cells in supercell of ZGNR-C/G, also applicable when creating heterojunctions (default: R=1)", metavar="[R]")
## option 5: toggle creation of finite structures
parser.add_argument("-finite", help="Set structure type to finite molecule instead or periodic (default: periodic)", action='store_true')
## option 6: choose if atoms should be displaced oop randomly
parser.add_argument("-oop", nargs=1, type=int, help="Activate random oop displacement, give number of structures to create (default: False)", metavar="[oop]")
## option 7: decide if .xyz file has 1D or 3D periodicity
parser.add_argument("-make1d", help="Deactivate VEC2 and VEC3 lines for .xyz file (default: False -> 3D system)", action='store_true')
## option 8: toggle saturation of dangling bonds
parser.add_argument("-saturate", help="Toggle saturation of dangling bonds with H (default: False -> unsaturated)", action='store_true')
## option 9: suppress any console output
parser.add_argument("-noprint", help="Add this command to suppress console outputs except for error messages (default: False)", action='store_true')

### parse arguments
args = parser.parse_args()

### interpret commands, apply default values
## option 9: suppress print commands if wanted (checked first as otherwise toggled too late)
if args.noprint: # command used
    sys.stdout = open(os.devnull, 'w')
## option 1: ribbon width
global N # ribbon width
try: 
    N = args.N[0]
    if N<=2: sys.exit(f"INPUT ERROR: ribbon width of N={N} given, needs to be at least 3!") # N<=2 can give disconnected molecules instead of GNR
    print('Ribbon width N:', N)
except: 
    sys.exit('INPUT ERROR: No ribbon width N specified!')
## identify requested structure type
global a # primitive unit cell length = distance between coves/gulfs on same edge
global b # cove/gulf offset between opposite edges
global M # gulf size
global struct_type # distinguish ZGNR-C and ZGNR-G
try: # check if ZGNR-C
    [a, b] = args.C # check if parameters given
    struct_type = 'cove' # set structure type to ZGNR-C
except: # check for other structure types
    try: # check if gulf
        [a, b, M] = args.G # check if parameters given
        struct_type = 'gulf' # set structure type to ZGNR-G
    except: # ZGNR ribbon needed
        struct_type = 'ZGNR' # set structure type to ZGNR
## option 2: parameters for ZGNR-C (cove-type) => sanity check of inputs
if struct_type=='cove':
    a, b = int(a), float(b) # transform data types
    # sanity check 1: b for odd vs. even N
    if N%2==0 and b%1!=0.0: sys.exit(f"VALUE ERROR: b needs to be integer for even N but is {b}!") # N is even: b integer
    if N%2!=0 and b%1!=0.5: sys.exit(f"VALUE ERROR: b needs to be half integer for odd N but is {b}!") # N is odd: b integer + 0.5
    # sanity check 2: a>=2 needed for cove-edges
    if a<2: sys.exit(f"VALUE ERROR: value of {a} given for a, but needs to be at least 2 in ZGNR-Cs!")
    # console output: name of structure
    if b.is_integer(): print(f"Structure: {N}-ZGNR-C({a},{int(b)})") # for integer b
    else: print(f"Structure: {N}-ZGNR-C({a},{b})") # for half integer b
## option 3: parameters for ZGNR-G (gulf-type) => sanity check of inputs
if struct_type=='gulf':
    a, b, M = int(a), float(b), int(M) # transform data types
    # sanity check 1: b for odd vs. even N
    if N%2==0 and b%1!=0.0: sys.exit(f"VALUE ERROR: b needs to be integer for even N but is {b}!") # N is even: b integer
    if N%2!=0 and b%1!=0.5: sys.exit(f"VALUE ERROR: b needs to be half integer for odd N but is {b}!") # N is odd: b integer + 0.5
    # sanity check 2: a>=M+1 required
    if a<M+1: sys.exit(f"VALUE ERROR: a needs to be at least {M+1} for ZGNR-Gs with gulfs of size M={M}!")
    # sanity check 3: M>1, if M=1 needs to be created ZGNR-C
    if M<1: sys.exit(f"VALUE ERROR: M needs to be at least 1 but is {M}!")
    if M==1: sys.exit(f"VALUE ERROR: M={M} is no ZGNR-G, ZGNR-C needs to be created instead!")
    # console output: name of structure
    if b.is_integer(): print(f"Structure: {N}-ZGNR-G{M}({a},{int(b)})") # for integer b
    else: print(f"Structure: {N}-ZGNR-G{M}({a},{b})") # for half integer b
## additional sanity check for both ZGNR-C and ZGNR-G: allowed value range of b
if struct_type!='ZGNR': # check for ZGNR-C/G
    if b<0 or b>a: sys.exit(f"INPUT ERROR: value of {b} for b outside primitive cell!")
    elif b>(a/2): sys.exit(f"VALUE ERROR: value of {b} for b outside allowed interval [0, a/2={a/2}]!")    
## option 4: number of repetition units of ZGNR-C/G (bulk in heterojunctions)
global nr_prim # number of primitive unit cells
try: # check if values are given
    nr_prim = args.R[0]  # asign value to new variable
    if nr_prim<1: sys.exit(f"INPUT ERROR: Impossible value of {nr_prim} for repeating units given. Needs to be integer with >=1!")
except: # no value given, option not activated
    nr_prim = 1 # set to 1 (default)
print(f"Primitive unit cells: {nr_prim}")
## option 5: toggle creation of finite structures
global periodicity # boolean if structure is periodic or not
if args.finite: # command used to make finite
    periodicity = 'finite' # finite molecule
else: # command not given = periodic system
    periodicity = 'periodic' # periodic model used
print(f"Periodicity: {periodicity}")
## option 6: choose if atoms should be displaced oop randomly
global do_oop # boolean to activate this option
global num_oop # number of structures to write
try: # oop specified?
    num_oop = args.oop[0] # get number of structures that have to be created
    do_oop = True # activate oop displacement routines in code
except: do_oop = False # no oop deformation performed
## option 7: decide if .xyz file has 1D or 3D periodicity
global make_1d # boolean to toggle between 1D and 3D unit cell
if args.make1d: 
    make_1d = True
    print('Cell dimensions: 1D')
else: 
    make_1d = False # default case
    print('Cell dimensions: 3D')
## option 8: toggle saturation of dangling bonds
global do_saturation # boolean for saturating dangling bonds
if args.saturate: # command given?
    do_saturation = True
    print("C atoms will be saturated with H!")
else: do_saturation = False
    
    
#=============================================#
# Auxiliary function: saturate dangling bonds #
#=============================================#

'''
* split off from workflow for better readability of code + routine called multiple times
* function allows distinction between periodic and finite structures from global parameters
* when multiple coordinate arrays are needed (backbone!) the numpy function np.concatenate((array1, array2, ...), axis=0) is used
* requires only coordinates of C atoms and unit cell information
* returns numpy array with C atom coordinates, numpy array with H atom coordinates and number of C atoms to remove
'''

def Saturate(coord, cell):
    
    ### prepare variables
    coord_H = [] # empty list to store H coordinates in
    C_to_remove = 0 # keep track if atoms have to be removed due to becoming isolated at termination
    
    ### saturate periodic structures
    if periodicity=='periodic':
        ## 3x1 supercell of atomic coordinates needed for saturation to avoid issues with PBC
        coord_sc = []  # supercell coordinate list
        for i in range(len(coord)): # over all C atoms
            coord_sc.append(coord[i]) # original structure
            coord_sc.append(coord[i] + cell[0]) # + x-vector of supercell
            coord_sc.append(coord[i] - cell[0]) # - x-vector of supercell
        coord_sc = np.asarray(coord_sc) # make numpy array
        ## create KDTree from supercell to find neighbors of C in original cell
        x_coord, y_coord, z_coord = coord_sc[:,0], coord_sc[:,1], coord_sc[:,2] # slice into coordinates
        tree = spatial.KDTree(list(zip(x_coord.ravel(), y_coord.ravel(), z_coord.ravel()))) # KDTree for neighbor search, uses only right part of cell
        ## find neighbors for all C atoms
        neigh = [] # store indices of neighbors
        for i in range(len(coord)): # check for each original atom
            neigh_here = tree.query_ball_point(coord[i], 1.1*DIST_CC) # search first neighbors
            neigh_here.remove(3*i) # remove atom itself, use "i*3" as 3x1 supercell was used
            neigh.append(neigh_here) # add to list 
        ## do saturation
        for i in range(len(coord)): # over all C atoms in original cell
            if len(neigh[i])==3: pass # saturated, do nothing
            elif len(neigh[i])==2: # one H needed to saturate
                neigh1, neigh2 = coord_sc[neigh[i][0]], coord_sc[neigh[i][1]] # get positions of neighbors
                vec_CC = neigh1 + neigh2 - 2*coord[i] # vector from C to its C neighbors summed up
                vec_CH = - vec_CC / np.sqrt(np.sum(vec_CC**2)) * DIST_CH # opposite direction for C-H bond, normalize and rescale by C-H bond length
                coord_H.append(coord[i] + vec_CH) # get H atom positions
            elif len(neigh[i])==1: # two H missing - this should not happen in periodic structures!
                sys.exit('ROUTINE ERROR: C atom with only one neighbor encountered in periodic system - this should never happen, check structure!')
            else:
                sys.exit('ROUTINE ERROR: isolated C atom encountered - this should never happen, check structure!')
    
    ### saturate finite molecules (minor changes compared to periodic ones, but kept separate for readability)
    if periodicity=='finite':
        ## create KDTree to find neighbors of C
        x_coord, y_coord, z_coord = coord[:,0], coord[:,1], coord[:,2] # slice into coordinates
        tree = spatial.KDTree(list(zip(x_coord.ravel(), y_coord.ravel(), z_coord.ravel()))) # KDTree for neighbor search, uses only right part of cell
        ## find neighbors for all C atoms
        neigh = [] # store indices of neighbors
        for i in range(len(coord)):
            neigh_here = tree.query_ball_point(coord[i], 1.1*DIST_CC) # search first neighbors
            neigh_here.remove(i) # remove atom itself, no "i*3" needed as finite structure!
            neigh.append(neigh_here) # add to list 
        ## do saturation
        for i in range(len(coord)): # over all C atoms in original cell
            if len(neigh[i])==3: pass # saturated, do nothing
            elif len(neigh[i])==2: # one H missing to saturate
                neigh1, neigh2 = coord[neigh[i][0]], coord[neigh[i][1]] # get positions of neighbors
                vec_CC = neigh1 + neigh2 - 2*coord[i] # vector from C to its C neighbors summed up
                vec_CH = - vec_CC / np.sqrt(np.sum(vec_CC**2)) * DIST_CH # opposite direction for C-H bond, normalize and rescale by C-H bond length
                coord_H.append(coord[i] + vec_CH) # get H atom positions
            elif len(neigh[i])==1: # two H missing to saturate
                neigh1 = coord[neigh[i][0]] # position of only neighbor
                vec_CC = neigh1 - coord[i] # vector of C-C bond
                # rotate vector by +/-120° using in-plane rotation matrix
                vec_CH_1 = np.array([vec_CC[0]*COS120-vec_CC[1]*SIN120, vec_CC[0]*SIN120+vec_CC[1]*COS120, vec_CC[2]])  # rotate by +120°
                vec_CH_2 = np.array([vec_CC[0]*COS120+vec_CC[1]*SIN120,-vec_CC[0]*SIN120+vec_CC[1]*COS120, vec_CC[2]])  # rotate by -120°
                # normalize to C-H distance, then add to H coordinates
                vec_CH_1 = vec_CH_1 / np.sqrt(np.sum(vec_CH_1**2)) * DIST_CH
                vec_CH_2 = vec_CH_2 / np.sqrt(np.sum(vec_CH_2**2)) * DIST_CH
                coord_H.append(coord[i] + vec_CH_1) # get H atom positions
                coord_H.append(coord[i] + vec_CH_2) # get H atom positions
            elif len(neigh[i])==0: # accidential isolated C: prepare to be removed in export phase
                coord[i][2] = -100 # shift away to detect for removal  
                C_to_remove += 1 # keep track how many C to remove
            else: # this case should never be triggered
                sys.exit('ROUTINE ERROR: invalid structure encountered while exporting finite structure - check input!')
        if C_to_remove>0:
            print(f"Saturation: {C_to_remove} isolated C atoms will be removed")
 
    ### return statement
    return coord, coord_H, C_to_remove


#=============================================#
# Auxiliary functions: export final structure #
#=============================================#
     
'''
* available formats: xsf and xyz, toggle when calling main export funtion
* xsf files are only created for periodic systems (better for visualization in VESTA), not created for finite molecules
* if no H atoms are needed: pass None object for coord_H when calling main export function
* main export routines includes option for oop deformation of C coordinates if requested => displacements handed over by separate list
* file name is finalized based on what is given to function
* main export function has no return values as its purpose is to write the structure file
'''

### function for export of atomic coordinates in xyz format => used for periodic and finite structures
def Coord_xyz(coord_C, coord_H, oop_C, oop_H):
    ## get number of atoms
    Z_C = len(coord_C) # do not include C_to_remove here as this would cause issues with writing to string
    if coord_H!=None: Z_H = len(coord_H) # only if H atoms are present
    else: Z_H = 0
    ## preamble of coordinates
    string = f"{Z_C + Z_H - C_to_remove}\n" # number of atoms, include that some C atoms might have to be removed
    string += "\n" # empty comment line
    ## write C atoms
    for i in range(Z_C):
        if coord_C[i][2]!=-100: # make sure atom was not labelled as ready to be removed by having z=-100
            string += f"C   {coord_C[i][0]:12.8f}   {coord_C[i][1]:12.8f}   {coord_C[i][2]+oop_C[i]:12.8f}\n"
    ## write H atoms
    if Z_H>0:
        for i in range(Z_H):
            string += f"H   {coord_H[i][0]:12.8f}   {coord_H[i][1]:12.8f}   {coord_H[i][2]+oop_H[i]:12.8f}\n"
    ## return statement
    return string # return coordiantes ready to write to file

### function for export of atomic coordinates in xsf format => will only be used for periodic structures
def Coord_xsf(coord_C, coord_H, oop_C, oop_H):
    ## get number of atoms
    Z_C = len(coord_C) # do not include C_to_remove here as this would cause issues with writing to string
    if coord_H!=None: Z_H = len(coord_H) # only if H atoms are present
    else: Z_H = 0
    ## preamble of coordinates
    string = 'PRIMCOORD\n'
    string += f"{Z_C + Z_H - C_to_remove} 1\n" # number of atoms, needs '1' afterwards to work due to format requirements
    ## write C atoms
    for i in range(Z_C):
        if coord_C[i][2]!=-100: # make sure atom was not labelled as ready to be removed by having z=-100
            string += f"6   {coord_C[i][0]:12.8f}   {coord_C[i][1]:12.8f}   {coord_C[i][2]+oop_C[i]:12.8f}\n"
    ## write H atoms
    if Z_H>0:
        for i in range(Z_H):
            string += f"1   {coord_H[i][0]:12.8f}   {coord_H[i][1]:12.8f}   {coord_H[i][2]+oop_H[i]:12.8f}\n"
    # return statement
    return string

### function for export of cell information in xyz format
def Cell_xyz(cell):
    ## write 1D unit cell
    string = ''
    string += f"VEC1   {cell[0][0]:12.8f}   {cell[0][1]:12.8f}   {cell[0][2]:12.8f}\n"
    ## if 3D unit cell is requested (default): write other vectors as well    
    if not make_1d:
        string += f"VEC2   {cell[1][0]:12.8f}   {cell[1][1]:12.8f}   {cell[1][2]:12.8f}\n"
        string += f"VEC3   {cell[2][0]:12.8f}   {cell[2][1]:12.8f}   {cell[2][2]:12.8f}\n"
    ## return statement
    return string

### function for export of cell information in xsf format
def Cell_xsf(cell):
    ## write string, always handled as 3D in xsf (my favourite visualizer VESTA doesn't accept lower periodicities)
    string = 'CRYSTAL\n'
    string += 'PRIMVEC\n'
    string += f"   {cell[0][0]:12.8f}   {cell[0][1]:12.8f}   {cell[0][2]:12.8f}\n" # 1st vector
    string += f"   {cell[1][0]:12.8f}   {cell[1][1]:12.8f}   {cell[1][2]:12.8f}\n" # 2nd vector
    string += f"   {cell[2][0]:12.8f}   {cell[2][1]:12.8f}   {cell[2][2]:12.8f}\n" # 3rd vector
    ## return statement
    return string

### main export function: actually create files
def WriteStructure(filename, coord_C, coord_H, cell, C_to_remove, write_xyz, write_xsf):
    ## prepare oop deformation even when it is not needed - done to simplify writing procedure
    # get number of atoms
    Z_C = len(coord_C) # do not include C_to_remove here as this would cause issues with writing to string
    if coord_H!=None: Z_H = len(coord_H) # only if H atoms are present
    else: Z_H = 0
    # initialize numpy arrays for displacement
    oop_C = np.zeros(Z_C, dtype=float)
    oop_H = np.zeros(Z_H, dtype=float)         
    ## write files without oop deformation
    if not do_oop:
        # write xyz file
        if write_xyz:
            file_xyz = open(filename+'.xyz', 'w+') # create file
            string = Coord_xyz(coord_C, coord_H, oop_C, oop_H) # write coordinates to string
            if periodicity=='periodic': # add cell information if periodic
                string += Cell_xyz(cell) # add cell information to string
            file_xyz.write(string) # write to file
            file_xyz.close() # finished
        # write xsf file
        if write_xsf:
            file_xsf = open(filename+'.xsf', 'w+') # create file
            string = Cell_xsf(cell) # write cell information (always needed for xsf)
            string += Coord_xsf(coord_C, coord_H, oop_C, oop_H) # write coordinates
            file_xsf.write(string) # write to file
            file_xsf.close() # finished     
    ## write files with oop deformation
    if do_oop:
        print('Adding oop displacement:')
        # create directory to store files inside as multiple are written for each structure
        os.mkdir(filename) # create directory to store all structures inside
        os.chdir(filename) # move into directory to create structures there
        # create as many oop displaced files as requested, use i_oop for file names
        for i_oop in range(1,num_oop+1): # amount taken from command line argument
            print(f"   {i_oop} of {num_oop}")
            # create displacements - done first to keep consistent for both xyz and xsf file
            for i in range(len(oop_C)):
                oop_C[i] = random.uniform(-OOP_MAX, OOP_MAX)
            for i in range(len(oop_H)):
                oop_H[i] = random.uniform(-OOP_MAX, OOP_MAX)
            # create xyz file
            if write_xyz:
                file_xyz = open(str(i_oop)+'.xyz', 'w+') # create file
                string = Coord_xyz(coord_C, coord_H, oop_C, oop_H) # write coordinates to string
                if periodicity=='periodic': # add cell information if periodic
                    string += Cell_xyz(cell) # add cell information to string
                file_xyz.write(string) # write to file
                file_xyz.close() # finished
            # write xsf file
            if write_xsf:
                file_xsf = open(str(i_oop)+'.xsf', 'w+') # create file
                string = Cell_xsf(cell) # write cell information (always needed for xsf)
                string += Coord_xsf(coord_C, coord_H, oop_C, oop_H) # write coordinates
                file_xsf.write(string) # write to file
                file_xsf.close() # finished
        # back to original working directory when everything is done for current structure
        os.chdir('..')
        
    
#============================#
# Create primitive unit cell #
#============================#

### set up matrix of C atoms for ZGNR without edge pattern
## how many rows of C atoms are in the primitive unit cell?
if struct_type=='ZGNR':
    C_prim = 2 # primitive unit cell
else: # cove-/gulf-edged ZGNR
    C_prim = 2*a # number of columns is 2a
## create matrix
C = np.ones((N, C_prim), dtype=int) # C[width, length] = C[rows, columns], all 1 = ZGNR ribbon

### modify ZGNR edge (if requested)
## add cove edges
if struct_type=='cove':
    C[N-1, 0] = 0 # first cove on top
    C[0, int(2*b)%(2*a)] = 0 # second cove on bottom of ribbon, make modulo 2a so b>=a is back in cell
## add gulf edges
if struct_type=='gulf':
    for i in range(2*M-1): # remove atoms to make gulf edge (cove edge is i=0)
        C[N-1, i] = 0 # create gulf edge on top
        position_bottom = int(2*b)%(2*a)+i # determine column position on bottom
        position_bottom_corrected = position_bottom%C_prim # fix to avoid issues when gulf crosses PBC
        C[0, position_bottom_corrected] = 0 # create gulf-edge on bottom
    # check if application of PBC is needed to determine position of inversion centers
    gulf_needs_PBC = False # default case: not needed
    if int(2*b)%(2*a)+2*M-1>=C_prim: gulf_needs_PBC = True # check if last position on bottom crosses PBC 
## calculate number of C atoms
Z_prim = np.count_nonzero(C) # in primitive cell
Z_C = np.count_nonzero(C)*nr_prim # in supercell

### determine C coordinates (and positions of CH vacancies)
C_coord = np.zeros((Z_C, 3), dtype=float) # C coordinates: first block is primitive unit cell, super cell is added afterwards (left empty until later)
vacancy_coord = [] # removed CH groups in primitive cell
## iterate over occupation matrix to get C coordinates
C_count, vacancy_count = 0, 0 # auxiliary variables
for row in range(0, N): # over rows
    for column in range(0, C_prim): # over columns of primitive cell
        # calculate coordinates
        x = column*COS30*DIST_CC # x coordinate
        y = (row + (row + (row+column+N%2)%2)*SIN30)*DIST_CC # y coordinate, N%2 added to adapt to have first cove on top instead of bottom of ribbon
        # move coordinates into middle of y- and z-vector
        y += 0.5*VACUUM
        z = 0.5*VACUUM
        # add C atom coordinates to their list
        if C[row, column]>0: # calculate coordinates if atom exists at this position
            C_coord[C_count] = [x, y, z] # save coordinates
            C_count += 1 # proceed to next C atom
        else: # no C atom written
            vacancy_coord.append([x, y, z])

### define primitive unit cell vector lengths
vec1 = C_prim*COS30*DIST_CC                             # x-direction (periodic direction of ribbon)
vec2 = max(C_coord[:,1]) - min(C_coord[:,1]) + VACUUM   # y-direction (ribbon is non-periodic)
vec3 = VACUUM                                           # z-direction (ribbon is non-periodic)


#============================#
# Identify inversion centers #
#============================#

'''
* First inversion center: between reference edge on top and first on bottom (the one shifted by b)
* Second inversion center: move into x-direction by half a primitive cell vector
* Nomenclature for file name handled so that inversion center on boundary of cell is used (not the one in the middle!)
'''

### find middle of ribbon in y-direction
y_center = 0
for i in range(len(vacancy_coord)): # over all C vacancies
    y_center += vacancy_coord[i][1]/len(vacancy_coord) 

### sort vacancies by top and bottom of ribbon
vacancies_list_top, vacancies_list_bottom = [], [] # sort coordinates
for i in range(len(vacancy_coord)): # over all C vacancies
    if vacancy_coord[i][1]<y_center: # check if bottom of ribbon
        vacancies_list_bottom.append(vacancy_coord[i]) # vacancy on bottom
    else: # position>=y_center
        vacancies_list_top.append(vacancy_coord[i]) # vacancy on top

### avoid bugs when PBC is applied which leads to wrong positioning when bottom gulf edge crosses boundary
if struct_type=='gulf' and gulf_needs_PBC: # fix only needed for ZGNR-G when b gets too large
    index = 0 # auxiliary variable, used to iterate over bottom row of ribbon
    vacancies_list_bottom = [] # overwrite vacancy positions on bottom for gulfs
    while C[0, index]==0: # do for bottom row until a carbon atom is reached
        x = index*COS30*DIST_CC + vec1 # x coordinate, moved by one unit cell to include PBC (adapted line!)
        y = ((index+N%2)%2)*SIN30*DIST_CC + 0.5*VACUUM # y coordinate (copied from above)
        z = 0.5*VACUUM # z coordinate (copied from above)
        vacancies_list_bottom.append([x, y, z]) # add to list of vacancies
        index += 1 # check next position
    for i in range(index, C_prim): # check remaining positions on bottom
        if C[0, i]==0: # only do when position is vacancy
            x = i*COS30*DIST_CC # x coordinate, no need to alter positions (copied from above)
            y = ((i+N%2)%2)*SIN30*DIST_CC + 0.5*VACUUM # y coordinate (copied from above)
            z = 0.5*VACUUM # z coordinate (copied from above)
            vacancies_list_bottom.append([x, y, z]) # add to list of vacancies

### determine center of vacancy at top and bottom
vacancy_top, vacancy_bottom = [0, 0, 0], [0, 0, 0] # split into new variables to calculate inv1 and inv2
for i in range(len(vacancies_list_top)): # assumes same number on top and bottom of ribbon
    for k in range(3): # do for x, y and z components
        vacancy_top[k] += vacancies_list_top[i][k]/len(vacancies_list_top) # sum up coordinates, normalize by number of points
        vacancy_bottom[k] += vacancies_list_bottom[i][k]/len(vacancies_list_bottom) # sum up coordinates, normalize by number of points
vacancy_top = np.asarray(vacancy_top) # make numpy array
vacancy_bottom = np.asarray(vacancy_bottom) # make numpy array

### calculate position of inversion centers
if struct_type=='ZGNR': # ZGNR
    if N%2==0: # even N: centers on boundary and middle of unit cell
        inv1 = np.array([0, vec2/2, vec3/2])
        inv2 = np.array([vec1/2, vec2/2, vec3/2])
    else: # odd N: move centers by 1/4 lattice vector to adapt to different edge geometry in odd N
        inv1 = np.array([vec1/4, vec2/2, vec3/2])
        inv2 = np.array([3*vec1/4, vec2/2, vec3/2])
else: # ZGNR-C/G
    inv2 = (vacancy_top+vacancy_bottom)/2 
    inv1 = inv2 + np.array([vec1/2, 0, 0])


#==================================#
# Create different unit cell types #
#==================================#    
        
'''
* Cells are created with 60°, 90°, and 120° angle between first two vectors
* This allows creation of different unit cells to match application and more flexibility when building heterojunctions
* Three types of structure at unit cell boundary are then possible: armchair, zigzag, and bearded
'''

### define cell types
cell_90 = np.array([[vec1, 0, 0], [0, vec2, 0], [0, 0, vec3]])
cell_60 = np.array([[vec1, 0, 0], [vec2/SIN60*COS60, vec2, 0], [0, 0, vec3]])
cell_120 = np.array([[vec1, 0, 0], [vec2/SIN120*COS120, vec2, 0], [0, 0, vec3]])    

### create lists for easier iteration
list_centers = [['inv1', inv1], ['inv2', inv2]] # inversion centers     
if struct_type=='ZGNR':
    list_cells = [['cell90', cell_90]] # only do 90° cell for ZGNR
else: # ZGNR-C/G
    list_cells = [['cell90', cell_90], ['cell60', cell_60], ['cell120', cell_120]] # use all three unit cell types for ZGNR-C/G

### iterate over unit cells and different inversion centers on boundary of cell
systems_for_export = [] # save systems to handle them later one by one
for element_cell in list_cells: # over all cell types
    label_cell = element_cell[0] # get label
    cell = element_cell[1] # get cell data
    cell_middle = (cell[0]+cell[1]+cell[2])/2 # middle of primitive unit cell in x,y-plane
    for element_center in list_centers: # over all inversion centers
        ## assign correct label to inversion center used
        if struct_type!='ZGNR': # ZGNR-C/G
            if element_center[0]=='inv1': label_center='S'      # inv1 = short point "S" on boundary
            elif element_center[0]=='inv2': label_center='L'    # inv2 = long point "L" on boundary 
        else: # ZGNR
            label_center = element_center[0] # use "inv1/2" labels
        ## adapt structure to match 
        center = element_center[1] # get inversion center position
        C_coord_local = copy.deepcopy(C_coord) # local copy of C coordinate list
        C_coord_local += cell_middle - center # move inversion center to middle of unit cell
        ## enforce periodic boundary conditions
        for i in range(len(C_coord_local)): # over all atoms
            x_new = np.linalg.solve(np.matrix.transpose(cell), C_coord_local[i]) # get relative coordinates, transposed cell needed for this procedure
            for j in range(3):
                while x_new[j]<0: x_new[j] += 1
                while x_new[j]>=1: x_new[j] -= 1
            x_new = np.dot(x_new, cell) # transform back to absolute coordinates
            C_coord_local[i] = copy.deepcopy(x_new) # overwrite coordinate with adapted value
        ## save needed information for further handling
        systems_for_export.append([C_coord_local, cell, label_cell, label_center])

### iterate over requested systems
for coord, cell, label_cell, label_center in systems_for_export:

    
    #=========================#
    # Detect termination type #
    #=========================#
    
    ### split cell artificially into left and right part by x_frac < or > 0.5
    coord_left, coord_right = [], [] # left and right part of unit cell
    for i in range(Z_prim): # over all atoms in primitive cell (first block of coordinates)
        x_new = np.linalg.solve(np.matrix.transpose(cell), coord[i]) # transposed cell needed for this procedure
        if x_new[0]<0.5: # left side of primitive unit cell
            coord_left.append(np.ndarray.tolist(np.dot(x_new, cell)))
        elif x_new[0]>=0.5 and x_new[0]<1.0: # right part primitive unit cell
            coord_right.append(np.ndarray.tolist(np.dot(x_new, cell)))
    coord_left, coord_right = np.asarray(coord_left), np.asarray(coord_right) # make numpy arrays
    
    ### use splitted coordinates to detect termination type
    ## move right half of atoms one unit cell to the left to check connectivity across boundary of unit cell
    x_coord, y_coord, z_coord = coord_right[:,0], coord_right[:,1], coord_right[:,2] # slice into coordinates
    x_coord -= cell[0][0] # shift by periodic unit cell vector in x-direction
    y_coord -= cell[0][1] # shift by periodic unit cell vector in y-direction
    z_coord -= cell[0][2] # shift by periodic unit cell vector in z-direction
    ## create KDTree for neighbor search, using only right part of cell
    tree = spatial.KDTree(list(zip(x_coord.ravel(), y_coord.ravel(), z_coord.ravel())))
    ## search for bonds across boundary of unit cell (left part with right part)
    pbc_bonds = np.zeros(len(coord_left), dtype=int) # count number of bonds which need PBC
    for i in range(len(coord_left)): # over all C atoms in left cell
        neigh_list = tree.query_ball_point(coord_left[i], 1.1*DIST_CC) # search first neighbors
        pbc_bonds[i] = len(neigh_list) # get number of first neighbors
    
    ### identify termination type based on results
    if label_cell=='cell90': 
        termination = 'armchair' # 90° cell always armchair
    elif max(pbc_bonds)==2: 
        termination = 'bearded' # atoms at bearded terminus have 2 PBC-bonds
    elif max(pbc_bonds)==1: 
        termination = 'zigzag' # atoms at zigzag terminus have only 1 PBC-bond
    else: 
        sys.exit('ROUTINE ERROR: termination could not be detected successfully!')
    
    
    #=============================#
    # Write supercell coordinates #
    #=============================#
    
    ### write coordinates previously left empty during creation of primitive unit cell
    for uc in range(1, nr_prim): # iterate over all additionally needed unit cells
        for atom in range(Z_prim): # over C atoms in primitive cell
            coord[atom+uc*Z_prim] = coord[atom] + uc*cell[0] # move by x-vector
    
    
    #===============================#
    # Create filename for exporting #
    #===============================#
    
    ### step-wise build filename from parameters
    ## ZGNR bulk
    filename = f"{N}-ZGNR"
    ## ZGNR-C/G
    if struct_type=='cove': # ZGNR-C
        if b.is_integer(): filename += f"-C_{a}_{int(b)}" # int() as b imported as float, only needed for even N
        else: filename += f"-C_{a}_{b}"
    if struct_type=='gulf': # ZGNR-G
        if b.is_integer(): filename += f"-G{M}_{a}_{int(b)}" # int() as b imported as float, only needed for even N
        else: filename += f"-G{M}_{a}_{b}"
    ## unit cell information
    filename += f"_{label_center}"  # add inversion center
    filename += f"_R{nr_prim}"      # add number of UC in supercell of bulk
    filename += f"_{label_cell}"    # add cell type
    filename += f"_{termination}"   # termination type
    ## periodicity and other post-processing information
    if periodicity=='finite': filename += '_finite'
    if periodicity=='periodic': filename += '_periodic'
    if do_saturation: filename += '_saturated'
    if do_oop: filename += '_oop'
    
    ### make variables needed for exporting global to avoid issues with local vs. global scope
    global C_to_remove
    C_to_remove = 0 # define once, updated later to match file requirements
    
    
    #===========================#
    # Saturate and export files #
    #===========================#
    
    ### create supercell from primitive unit cell
    cell_export = copy.deepcopy(cell) # prepare object for supercell
    cell_export[0] *= nr_prim # make supercell along ribbon as required
    ### finite structure without saturation: write xyz, no xsf needed
    if periodicity=='finite' and do_saturation==False:
        WriteStructure(filename, coord, None, cell_export, 0, True, False)
    ### periodic structure without saturation: write xyz and xsf
    if periodicity=='periodic' and do_saturation==False:
        WriteStructure(filename, coord, None, cell_export, 0, True, True)
    ### finite structure with saturation: write xyz, no xsf needed
    if periodicity=='finite' and do_saturation==True:
        coord_C, coord_H, C_to_remove = Saturate(coord, cell_export)
        WriteStructure(filename, coord_C, coord_H, cell_export, C_to_remove, True, False)
    ### periodic structure with saturation: write xyz and xsf
    if periodicity=='periodic' and do_saturation==True:
        coord_C, coord_H, C_to_remove = Saturate(coord, cell_export)
        WriteStructure(filename, coord_C, coord_H, cell_export, C_to_remove, True, True)


#==================#
# Finish iteration #
#==================#

print("Structure exported!")