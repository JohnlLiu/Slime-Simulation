import numpy as np
import os
import matplotlib.pyplot as plt
import random, math
import glob
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
import shutil


####Parameters####

#Environment Parameters:
MAP_SIZE = 300 #Square map of size nxn pixels
AGENTS_NUM = 250 #Number of agents to simulate in the map
CIRCLE_SIZE = 0.05 #Size of circle spawn as proportionate to map size
ITERATIONS = 500 #number of iterations to run
RENDER_SPEED = 1

#Agent Parameters
SPEED = 1 #speed of agents
TURN_STRENGTH = 0.35 #turning strength. Multiplied by turn angle to adjust turning degree

#Sensor Parameters
SEN_SIZE = 1 #size of neighbordhood around sensor position eg. sen_size=1 -> sensor size of 3x3
SEN_ANGLE = 45 #Offset angle of left and righ sensors (left is -SEN_ANGLE)
SEN_DIST = 6 #Offset distance of sensor

#Scent Parameters
TRAIL_STRENGTH = 1 #value layed by each agent
MAX_STRENGTH = 20 #max value of scent allowed on map
DECAY_RATE = 0.0015 #rate at which scent value decays
BLUR_STRENGTH = 0.5 #Strength of blur from gaussian filter (if being used)
DIFFUSE_WEIGHT = 0.3 #Rate at which scent diffuses

# How much time passes per frame
STEP_SIZE = 1


####Initialization####

texture_map = None
agents = None

def CircleSpawn_RandAng(agents, spawn_zone):

    for i in range(AGENTS_NUM):
        while True:
            x = random.randint(-spawn_zone, spawn_zone)
            y = random.randint(-spawn_zone, spawn_zone)
            if x**2 + y**2 > spawn_zone**2:
                continue
            agents[i, 0] = x + MAP_SIZE/2 #x position
            agents[i, 1] = y + MAP_SIZE/2 #y position
            agents[i, 2] = random.randint(0, 360) #random angle
            break
    
    return agents

def CircleSpawn_FaceCent(agents, spawn_zone):

    for i in range(AGENTS_NUM):
        while True:
            x = random.randint(-spawn_zone, spawn_zone)
            y = random.randint(-spawn_zone, spawn_zone)
            if x**2 + y**2 > spawn_zone**2:
                continue
            agents[i, 0] = x + MAP_SIZE/2 #x position
            agents[i, 1] = y + MAP_SIZE/2 #y position

            #spawn center
            xx = MAP_SIZE/2
            yy = MAP_SIZE/2

            agents[i, 2] = math.atan2((yy-agents[i, 0]),(xx-agents[i, 1]))*180/math.pi
            break
    
    return agents


def CircleSpawn_FaceOut(agents, spawn_zone):

    for i in range(AGENTS_NUM):
        while True:
            x = random.randint(-spawn_zone, spawn_zone)
            y = random.randint(-spawn_zone, spawn_zone)
            if x**2 + y**2 > spawn_zone**2:
                continue
            agents[i, 0] = x + MAP_SIZE/2 #x position
            agents[i, 1] = y + MAP_SIZE/2 #y position

            #spawn center
            xx = MAP_SIZE/2
            yy = MAP_SIZE/2

            agents[i, 2] = math.atan2((yy-agents[i, 0]),(xx-agents[i, 1]))*180/math.pi + 180
            break
    
    return agents

def RandomSpawn(agents):

    for i in range(AGENTS_NUM):
        x = random.randint(0, MAP_SIZE)
        y = random.randint(0, MAP_SIZE)
        agents[i, 0] = x
        agents[i, 1] = y
        agents[i, 2] = random.randint(0, 360) 

    return agents

def init_scene ():
    global texture_map
    global agents

    # Create the map
    texture_map = np.random.random((MAP_SIZE, MAP_SIZE))/100
    #texture_map = np.zeros((MAP_SIZE, MAP_SIZE))

    # Update circle size
    # Area on map where the agents spawn
    spawn_zone = int(CIRCLE_SIZE * MAP_SIZE)

    # Spawn agents within spawn area at random positions and angles
    agents = np.zeros((AGENTS_NUM, 3))
    
    agents = RandomSpawn(agents)


# Blurrs the map and then reduces the intensity of the map
def dissipate_trails (map) :

    blurred = uniform_filter(map, size=(3,3)) #median filter to decrease scent weights
    blurred = map * (1 - DIFFUSE_WEIGHT) + blurred * (DIFFUSE_WEIGHT) #rate at which scents dissapate
    trails = blurred - DECAY_RATE #rate at which scents decay
    trails[trails<0] = 0 #keep minimum values at 0
    map[:,:] = trails

    return trails


def sense(map, agents, sen_ang, sen_dist, sen_size):

    agent_pos = agents[:,:2]
    agent_ang = agents[:,2:]

    sens_direction = np.array(np.empty_like(agent_pos)).astype('float64')
    sens_direction[:,:1] = np.sin((agent_ang+sen_ang)*math.pi/180)
    sens_direction[:,1:] = np.cos((agent_ang+sen_ang)*math.pi/180)

    sen_pos = agent_pos+sens_direction*sen_dist
    sen_pos = sen_pos.astype('int64')

    strength = np.zeros((len(agents),1), dtype=float)

    for i in range(len(sen_pos)):
        x, y = sen_pos[i][0], sen_pos[i][1]

        tot_str = 0

        for n in range(-sen_size, sen_size+1):
            for m in range(-sen_size, sen_size+1):

                if n+x>=MAP_SIZE or m+y>=MAP_SIZE or n+x<0 or m+y<0: #sensor out of range of map
                    continue
                else:
                    tot_str+=map[x+n][y+m]
        
        strength[i] = tot_str

    return strength

# Moves the adgents along the map based on angle from sense function
def update_agents(_map, agents):

    #calculate sensor strengths for forward, left, and right sensors
    forward_sense = sense(_map, agents, 0, SEN_DIST, SEN_SIZE)
    left_sense = sense(_map, agents, -SEN_ANGLE, SEN_DIST, SEN_SIZE)
    right_sense = sense(_map, agents, SEN_ANGLE, SEN_DIST, SEN_SIZE)

    #store the change in angles according to sensor results
    turn_ang = np.zeros((len(agents),1))

    #amount of turn each agent is allowed to perform
    turn_speed = TURN_STRENGTH*360

    for i in range(len(agents)):

        randSteer = random.uniform(0,1)

        if forward_sense[i]>left_sense[i] and forward_sense[i]>right_sense[i]:
            turn_ang[i] = 0 #keep moving forward
        elif forward_sense[i]<left_sense[i] and forward_sense[i]<right_sense[i]:
            turn_ang[i] = (randSteer-0.5) * 2 * turn_speed #move randomly
        elif left_sense[i]>right_sense[i]:
            turn_ang[i] = -randSteer * turn_speed #turn towards the left
        elif right_sense[i]>left_sense[i]:
            turn_ang[i] = randSteer * turn_speed #turn towards the right

    # Get agent positions and orientation
    pos = agents[:,:2]
    ang = agents[:,2:] + turn_ang

    direction = np.array(np.empty_like(pos))

    # compute direction of each
    direction[:,:1] = np.sin(ang*math.pi/180)
    direction[:,1:] = np.cos(ang*math.pi/180)

    new_pos = pos+direction*SPEED

    #find if any agents are out of bounds
    x_oob = np.any([new_pos[:,0] < 0, new_pos[:,0] >= MAP_SIZE], axis=0)
    y_oob = np.any([new_pos[:,1] < 0, new_pos[:,1] >= MAP_SIZE], axis=0)

    id_oobx = []
    id_ooby = []

    for i in range(len(x_oob)):
        if x_oob[i] == True:
            if i not in id_oobx:
                id_oobx.append(i)

    for i in range(len(y_oob)):
        if y_oob[i] == True:
            if i not in id_ooby:
                id_ooby.append(i)


    #redirect any out of bounds agents back into area
    for j in id_oobx:
        new_pos[j][0] = min(MAP_SIZE-1,max(0, new_pos[j][0]))
        #agents[j][2] = np.random.randint(0,360)
        agents[j][2] = -agents[j][2] #bounce off boundary


    for k in id_ooby:
        new_pos[k][1] = min(MAP_SIZE-1,max(0, new_pos[k][1]))
        #agents[k][2] = np.random.randint(0,360)
        agents[k][2] = -agents[k][2]+180 #bounce off boundary


    agents[:,:2] = new_pos


def lay_trails (map, agents ):
    integer_positions = np.floor(agents[:,:2]).astype(int)
    map[integer_positions[:,0], integer_positions[:,1]] = TRAIL_STRENGTH


def simulation_step (map, agents):
    dissipate_trails(map)
    update_agents(map, agents)
    lay_trails(map, agents)


def show_map (m, save, do) :
    
    # copy the map to a new array and padd
    map = np.zeros((MAP_SIZE+4, MAP_SIZE+4))
    map[2:-2,2:-2] = m

    map = map / map.max() * 255
    colored = np.zeros((MAP_SIZE+4, MAP_SIZE+4))
    colored[:,:] = map
    colored =colored.astype(np.uint8)


    if not do: return

    # Render the map
    im = Image.fromarray(colored)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(f'./results/{save[0]}/saves/{save[1]}.png')



def run_simulation ():

    global texture_map

    # Setup Saving
    run_id = str(random.randint(0, 1000000))

    cur_dir = os.getcwd()
    new_dir = os.path.join(cur_dir, r'results', run_id, r'saves')
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    param_path = os.path.join(cur_dir, r'results', run_id, r'parameters.txt')

    f = open(param_path, 'w+')

    para_names = ["Agent Num: ", "Agent Speed: ", "Turn Strength: ", "Sensor Size: ", "Sensor Angle: ", "Sensor Distance: ",
                "Decay Rate: ", "Diffuse Rate: "]

    para_values = [AGENTS_NUM, SPEED, TURN_STRENGTH, SEN_SIZE, SEN_ANGLE, SEN_DIST, DECAY_RATE, DIFFUSE_WEIGHT]

    for line in range(len(para_names)):
        f.write(para_names[line])
        f.write(str(para_values[line]))
        f.write('\n')
    
    f.close()

    print(f'Saving as {run_id}')

    # Initialize the scene
    init_scene()

    # Run the simulation
    for i in range(ITERATIONS):
        print(f'{i+1}/{ITERATIONS}', end='\r')
        simulation_step(texture_map, agents)
        show_map(texture_map, (run_id, str.rjust(str(i),4,'0')), i % RENDER_SPEED == 0)

    return run_id


sim_id = run_simulation()


def compile_results ( _id ):

    # Load all the images
    images = glob.glob(f"./results/{sim_id}/saves/*.png")
    images.sort()

    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(f"./results/{_id}/out.gif", format="GIF", append_images=frames,
               save_all=True, duration=20, loop=0)


compile_results( sim_id )
print(sim_id)

##Courtesy to Eric Robert for parts of code for saving and compiling images as GIF:
##https://github.com/eric-robert/particle-simulator