#import matplotlib.pyplot as plt
import ravenui
import numpy as np
import pygame as pg
import noise, cv2 # pip3 install opencv-python
import random
vec = pg.math.Vector2
from PIL import Image

WIDTH = 800
HEIGHT = 650
FPS = 15
# initialize pg and create window
pg.mixer.pre_init(44100, -16, 1, 512) #reduces the delay in playing sounds
pg.init()
pg.mixer.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Map Generator")
clock = pg.time.Clock()

MAP_SIZE = 1000
TILE_COUNT = 798
TILESET_COLUMNS = 27
SHALLOW_TILE = 217
WATER_TILE = 218
SWAMP_SHALLOWS = 219
SWAMP_WATER = 220
LAKE_SHALLOWS = 221
LAKE_WATER = 222
RIVER_TILES = [224, 223, 225, 226]
RIVER_CHANCE = 5
RIVER_DIRCHANCE = 10
RIVER_ROCKS = [636, 637, 638, 639, 640]
WAVE_TILES = [231, 232, 233,258, 260, 285, 286, 287]
WAVE_CORNER_TILES = [237, 238, 480, 481]
WAVE_CORNER_TILES2 = [265, 264, 454, 453]
MAIN_TREE_TILES = [649, 656, 838, 845]
ALL_TREE_TILES = []
TREE_DIMENTIONS = 7
for tree in MAIN_TREE_TILES:
    tile = tree
    for y in range(0, TREE_DIMENTIONS):
        for x in range(0, TREE_DIMENTIONS):
            ALL_TREE_TILES.append(tile)
            tile += 1
        tile += (TILESET_COLUMNS - TREE_DIMENTIONS)
PALM_PROB = 10
PINE_PROB = 7
GREENTREE_PROB = 10
DEADTREE_PROB = 10
# Plant dictionaries are by tile and how common they are. Higher numbers are less common.
SWAMPW_PLANTS = {259: 1}
SWAMPD_PLANTS = {270: 3, 297: 1}
OCEAN_PLANTS = {262: 1}
MOUNTAIN_PLANTS = {424: 10, 505: 10, 269: 4}
FOREST_PLANTS = {340: 1, 586: 3, 505: 4, 421: 7}
GRASSLAND_PLANTS = {586: 1, 583: 15, 502: 30, 424: 25, 421: 20, 269: 7, 297: 3}
WASTELAND_PLANTS = {243: 20, 242: 10, 269: 8}
DESERT_PLANTS = {343: 1, 243: 50, 242: 150}
SPLANTW_FACT = 12
SPLANTD_FACT = 2
OPLANT_FACT = 30
MPLANT_FACT = 7
FPLANT_FACT = 10
GPLANT_FACT = 20
WPLANT_FACT = 25
DPLANT_FACT = 100

MAX_SWAMP_SIZE = 100 * 100 # Max area swamps can take up.
MIN_OCEAN_SIZE = 1000 * 40 # Minimum area needed to add ocean water
KINDSOFTILES = 18 #Ignoring the water tiles and swamp tiles
ROUND_TILES = range(89, 133)
ROUNDPOINT_TILES = range(155, 177)
MASK_FACTOR = 0
ELEVATION_FACTOR = 0
SEALVL_FACTOR = 1

BLACK = [0, 0, 0]
colors = [[93, 120, 150], [108, 135, 168], [147, 174, 171], [165, 195, 228], [189, 219, 219], [261, 231, 174], [152, 118, 84], [102, 159, 69], [75, 141, 33], [102, 66, 45], [120, 105, 72], [123, 108, 84], [144, 144, 144], [168, 168, 168], [183, 183, 183], [225, 225, 243], [255, 255, 255], [240, 250, 255]]
swampcolors = [[0, 105, 150], [0, 125, 130], [0, 145, 110], [0, 160, 100], [0, 165, 75], [0, 170, 55], [0, 180, 55], [0, 190, 55], [0, 205, 50], [0, 205, 50], [0, 205, 50], [0, 205, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
RIVER_COLOR = [0, 0, 255]

#newcolors = []
#for color in colors:
#    newcolor = []
#    for v in color:
#        v2 = int(v*3/2)
#        newcolor.append(v2)
#    newcolors.append(newcolor)
#print(newcolors)
threshold = 0
map_surface = pg.Surface((MAP_SIZE, MAP_SIZE)).convert()
scale_map_surface = pg.Surface((500, 500)).convert()
world_noise = np.zeros((MAP_SIZE, MAP_SIZE))
random_plant_noise = np.zeros((MAP_SIZE, MAP_SIZE))
world = np.zeros((MAP_SIZE, MAP_SIZE))
river_edges = np.zeros((MAP_SIZE, MAP_SIZE), int)
river_edges2 = np.zeros((MAP_SIZE, MAP_SIZE), int)
deep_river_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)
swamps_arr = np.zeros((MAP_SIZE, MAP_SIZE), int) # Used for storing regions that can be turned into swamps.
ocean_regions = np.zeros((MAP_SIZE, MAP_SIZE), int) # Used to tell oceans/seas from lakes to add waves to.
beach_regions = np.zeros((MAP_SIZE, MAP_SIZE), int) # Used to tell were the beaches are.
rivers = np.zeros((MAP_SIZE, MAP_SIZE), int) # Used for river layer.
img = cv2.imread("mapmask1.png", 0)
mask = img / 255.0

noise_type = noise.pnoise2
scale = 100.0
octaves = 6
persistence = 0.5
lacunarity = 2.0

def return_flags(bitnum): # Used for returning the rotational flags of a tile so it can be replaced by one of the same rotation.
    tileid = bitnum & ~(0x80000000 | 0x40000000 | 0x20000000)
    if tileid == bitnum & ~(0x20000000):
        return 0x20000000
    elif tileid == bitnum & ~(0x40000000):
        return 0x40000000
    elif tileid == bitnum & ~(0x80000000):
        return 0x80000000
    elif tileid == bitnum & ~(0x40000000 | 0x20000000):
        return 0x40000000 | 0x20000000
    elif tileid == bitnum & ~(0x80000000 | 0x20000000):
        return 0x80000000 | 0x20000000
    elif tileid == bitnum & ~(0x80000000 | 0x40000000):
         return 0x80000000 | 0x40000000
    else:
        return 0x80000000 | 0x40000000 | 0x20000000

def make_river(tbase, y, x, dr, count = 0):
    global rivers, world_noise, swamps_arr, river_edges, river_edges2
    print(world_noise[y, x])
    if (count < 900) and (tbase[y, x] > 5) and (swamps_arr[y, x] == 0):
        upv = world_noise[y - 1, x]
        dnv = world_noise[y + 1, x]
        ltv = world_noise[y, x - 1]
        rtv = world_noise[y, x + 1]
        cpv = world_noise[y, x]
        px = x
        py = y
        dirlist = [upv, dnv, ltv, rtv]
        xposlist = [x, x, x - 1, x + 1]
        yposlist = [y - 1, y + 1, y, y]
        minpos = dirlist.index(min(dirlist))
        x = xposlist[minpos]
        y = yposlist[minpos]
        if cpv < world_noise[y, x]: #Raises up current tile if it's a local minimum.
            world_noise[py, px] = world_noise[y, x] + 0.005
        if rivers[y, x] == 0:
            rivers[y, x] = RIVER_TILES[minpos]
        else:
            minpos = dirlist.index(sorted(dirlist)[1])
            x = xposlist[minpos]
            y = yposlist[minpos]
            if rivers[y, x] == 0:
                rivers[y, x] = RIVER_TILES[minpos]
            else:
                minpos = dirlist.index(sorted(dirlist)[2])
                x = xposlist[minpos]
                y = yposlist[minpos]
                if rivers[y, x] == 0:
                    rivers[y, x] = RIVER_TILES[minpos]
                else:
                    minpos = dirlist.index(sorted(dirlist)[3])
                    x = xposlist[minpos]
                    y = yposlist[minpos]
                    if rivers[y, x] == 0:
                        rivers[y, x] = RIVER_TILES[minpos]
        if minpos in [0, 1]:
            if river_edges[y, x + 1] == 0:
                river_edges[y, x + 1] = RIVER_ROCKS[3]
            else:
                river_edges2[y, x + 1] = RIVER_ROCKS[3]
            if river_edges[y, x - 1] == 0:
                river_edges[y, x - 1] = hflip(RIVER_ROCKS[3])
            else:
                river_edges2[y, x - 1] = hflip(RIVER_ROCKS[3])
        if minpos in [2, 3]:
            if river_edges[y + 1, x] == 0:
                river_edges[y + 1, x] = dflip(RIVER_ROCKS[3])
            else:
                river_edges2[y + 1, x] = dflip(RIVER_ROCKS[3])
            if river_edges[y - 1, x] == 0:
                river_edges[y - 1, x] = vdflip(RIVER_ROCKS[3])
            else:
                river_edges2[y - 1, x] = vdflip(RIVER_ROCKS[3])

        count += 1
        make_river(tbase, y, x, dr, count)


def ul_corner(tile):
    tile += TILESET_COLUMNS #Shifts tile ID down to corner row
    return tile
def ur_corner(tile):
    tile += TILESET_COLUMNS #Shifts tile ID down to corner row
    tile = tile | 0x40000000 # Vertical flip
    return tile
def ll_corner(tile):
    tile += TILESET_COLUMNS #Shifts tile ID down to corner row
    tile = tile | 0x80000000 #Horizontal flip
    return tile
def lr_corner(tile):
    tile += TILESET_COLUMNS #Shifts tile ID down to corner row
    tile = tile | 0x80000000 #Horizontal flip
    tile = tile | 0x40000000 #Vertical flip
    return tile
def pu_corner(tile):
    tile += TILESET_COLUMNS*2
    tile = tile | 0x20000000 #diagonal flip
    return tile
def pd_corner(tile):
    tile += TILESET_COLUMNS*2
    tile = tile | 0x20000000 #diagonal flip
    tile = tile | 0x80000000 #Horizontal flip
    return tile
def pr_corner(tile):
    tile += TILESET_COLUMNS*2
    tile = tile | 0x40000000 # Vertical flip
    return tile
def pl_corner(tile):
    tile += TILESET_COLUMNS*2
    return tile

def tl_corner(tile):
    tile += TILESET_COLUMNS*3
    tile = tile | 0x20000000 #diagonal flip
    return tile
def tr_corner(tile):
    tile += TILESET_COLUMNS*3
    tile = tile | 0x20000000 #diagonal flip
    tile = tile | 0x80000000 #Horizontal flip
    return tile
def td_corner(tile):
    tile += TILESET_COLUMNS*3
    tile = tile | 0x40000000 # Vertical flip
    return tile
def tu_corner(tile):
    tile += TILESET_COLUMNS*3
    return tile

def roundedpoint0(tile):
    tile += TILESET_COLUMNS*7
    tile = tile | 0x40000000 # Vertical flip
    return tile
def roundedpoint1(tile):
    tile += TILESET_COLUMNS*7
    return tile
def roundedpoint2(tile):
    tile += TILESET_COLUMNS * 7
    tile = tile | 0x80000000 #Horizontal flip
    tile = tile | 0x40000000 # Vertical flip
    return tile
def roundedpoint3(tile):
    tile += TILESET_COLUMNS*7
    tile = tile | 0x80000000 #Horizontal flip
    return tile

def nul_corner(tile):
    tile += TILESET_COLUMNS*6 #Shifts tile ID down to corner row
    return tile
def nur_corner(tile):
    tile += TILESET_COLUMNS*6 #Shifts tile ID down to corner row
    tile = tile | 0x40000000 # Vertical flip
    return tile
def nll_corner(tile):
    tile += TILESET_COLUMNS*6 #Shifts tile ID down to corner row
    tile = tile | 0x80000000 #Horizontal flip
    return tile
def nlr_corner(tile):
    tile += TILESET_COLUMNS*6 #Shifts tile ID down to corner row
    tile = tile | 0x80000000 #Horizontal flip
    tile = tile | 0x40000000 #Vertical flip
    return tile

def hflip(tile):
    tile = tile | 0x80000000  # Horizontal flip
    return tile
def vflip(tile):
    tile = tile | 0x40000000  # Vertical flip
    return tile
def hvflip(tile):
    tile = vflip(hflip(tile))
    return tile
def dflip(tile):
    tile = tile | 0x20000000  # diagonal flip
    return tile
def hdflip(tile):
    tile = dflip(hflip(tile))
    return tile
def vdflip(tile):
    tile = dflip(vflip(tile))
    return tile

def switch_noise():
    global noise_type, noise_txt
    if noise_type == noise.snoise2:
        noise_type = noise.pnoise2
        noise_txt = "Perlin Noise"
    elif noise_type == noise.pnoise2:
        noise_type = noise.snoise2
        noise_txt = "Simplex Noise"
    noise_button.update_text(noise_txt)

def add_color(world):
    global colors, KINDSOFTILES, swampcolors, swamps_arr
    color_world = np.zeros(world.shape+(3,))

    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            for t in range(KINDSOFTILES):
                if t == KINDSOFTILES-1:
                    if world[i][j] < threshold + 1:
                        color_world[i][j] = colors[t]
                elif world[i][j] < threshold + (t+1)/KINDSOFTILES:
                    if swamps_arr[i][j] == 1:
                        color_world[i][j] = swampcolors[t]
                    else:
                        color_world[i][j] = colors[t]
                    break
                elif world[i][j] >= threshold + 1:
                    color_world[i][j] = colors[KINDSOFTILES-1]
            if rivers[i][j]:
                color_world[i][j] = RIVER_COLOR
    return color_world

def new_map():
    global scale_map_surface, world, mask, swamps_mask, random_plant_noise
    "Building new map..."
    # Noise settings
    # Creates a perlin noise array the size of the map.
    world = np.zeros((MAP_SIZE, MAP_SIZE))
    rscale = random.randrange(40, 70)
    roct = random.randrange(4, 12)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            world[i][j] = noise_type(i/scale_slider.val,
                                        j/scale_slider.val,
                                        octaves=octaves_slider.val,
                                        persistence=persistence_slider.val,
                                        lacunarity=lacunarity_slider.val,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=0) + 0.5  # The 0.5 makes i return values between 0 and 1.
            random_plant_noise[i][j] = noise.snoise2(i/rscale,
                                        j/rscale,
                                        octaves=roct,
                                        persistence=0.5,
                                        lacunarity=2.0,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=0) + 0.5  # The 0.5 makes i return values between 0 and 1.
    max_grad = np.amax(world)
    world = world / max_grad
    max_grad = np.amax(random_plant_noise)
    random_plant_noise = random_plant_noise / max_grad
    random_plant_noise = np.where(random_plant_noise > 0.5, 1, 0) # Used to generate randomish plant grown patterns.
    update_map()

def update_map():
    global world_noise, scale_map_surface, world, mask, swamps_arr, MAX_SWAMP_SIZE, ocean_regions, beach_regions, rivers, deep_river_arr
    swamps_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used for storing regions that can be turned into swamps.
    snow_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)
    ocean_regions = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used to tell oceans/seas from lakes to add waves to.
    beach_regions = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used to tell were the beaches are.
    print('Updating map...')
    # Combines mask image data with perlin noise
    world_noise = np.zeros_like(world)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            world_noise[i][j] = ((world[i][j]+elev_slider.val) * (mask[i][j]+mask_slider.val))
    # get it between 0 and 1
    max_grad = np.amax(world_noise)
    world_noise = world_noise / max_grad
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if sealvl_slider.val != 1: #Changes sea level
                world_noise[i][j] *= sealvl_slider.val
    world_noise = np.where(world_noise < 0, 0, world_noise)  # Gets rid of negatives.
    world_noise = np.where(world_noise > 0.98, 0.98, world_noise)  # Gets rid of values above 1ish.

    # Figures out regions where swamps can be and stores each swamp region in a list.
    temp_array = np.array(world_noise)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            x = temp_array[i][j]
            newx = int(x * (KINDSOFTILES - 1))
            newx += 1  # Gets rid of zeros
            temp_array[i][j] = newx
            if newx == 17:
                snow_arr[i][j] = 1 # Marks potential spots for river heads in snowy areas.
    grass_arr = np.where(temp_array < 9, 255, 0)
    grass_img = grass_arr.astype('uint8')
    kernel = np.ones((3, 3), np.uint8)
    grass_img = cv2.dilate(grass_img, kernel, iterations=1) # This adds a padding of 1 around the swamp areas so the edges are correct.
    ocean_arr = np.where(temp_array < 6, 255, 0) # Used for finding where the oceans are.
    ocean_img = ocean_arr.astype('uint8')
    ocean_img = cv2.dilate(ocean_img, kernel, iterations=1)
    water_img = ocean_img.copy()
    water_regions = np.where(water_img > 0, 1, 0)
    shade = 0
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if grass_img[x, y] == 255:
                shade += 1
                if shade > 255: # Limited to 255 swamps
                    shade = 255
                    break
                cv2.floodFill(grass_img, None, (y, x), shade)
    swamps_list = []
    for color in range(0, shade):
        swamp_arr = np.where(grass_img == color, 1, 0)
        if np.count_nonzero(swamp_arr == 1) < MAX_SWAMP_SIZE:
            swamps_list.append(swamp_arr)

    shade = 0
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if ocean_img[x, y] == 255:
                shade += 1
                if shade > 255: # Limited to 255 swamps
                    shade = 255
                    break
                cv2.floodFill(ocean_img, None, (y, x), shade)

    # Finds ocean areas.
    for color in range(0, shade):
        ocean_arr = np.where(ocean_img == color, 1, 0)
        if np.count_nonzero(ocean_arr == 1) > MIN_OCEAN_SIZE:
            ocean_regions = np.add(ocean_regions, ocean_arr) # Finds the ocean tiles to add waves to
    ocean_regions = np.where(water_regions == 1, ocean_regions, 0) # Used for removing a bug introduced with floodFill. It removes all non water places from oceans.
    beach_arr = np.where(ocean_regions == 1, 255, 0)
    beach_img = beach_arr.astype('uint8')
    beach_regions = cv2.dilate(beach_img, kernel, iterations=20) # Used for palm trees and plant placement.
    beach_regions = np.where(beach_regions > 125, 1, 0)
    print("Potential swamp areas:" + str(len(swamps_list)))
    if len(swamps_list) < swamps_slider.val:
        num_swamps = len(swamps_list)
    else:
        num_swamps = swamps_slider.val
    swamps_list = random.sample(swamps_list, num_swamps) # Reduces number of swamps in list to up to the selected number.
    for swamp in swamps_list:
        swamps_arr = np.add(swamps_arr, swamp)

    # Creates rivers
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if snow_arr[i][j] == 1:
                count = [0, 0, 0, 0]
                # Marks first tiles for river heads then creates rivers where they flow to the ocean or lakes.
                if temp_array[i - 1][j] == (temp_array[i][j] - 1):
                    if random.randrange(0, RIVER_CHANCE) == 1:
                        make_river(temp_array, i, j, 1)
                if temp_array[i + 1][j] == (temp_array[i][j] - 1):
                    if random.randrange(0, RIVER_CHANCE) == 1:
                        make_river(temp_array, i, j, 0)
                if temp_array[i][j - 1] == (temp_array[i][j] - 1):
                    if random.randrange(0, RIVER_CHANCE) == 1:
                        make_river(temp_array, i, j, 3)
                if temp_array[i][j + 1] == (temp_array[i][j] - 1):
                    if random.randrange(0, RIVER_CHANCE) == 1:
                        make_river(temp_array, i, j, 2)

    ocean_regions = np.where(water_regions == 1, ocean_regions, 0) # Used for removing a bug introduced with floodFill. It removes all non water places from oceans.
    deep_river_arr = np.where(rivers > 0, 255, 0)
    deep_river_img = deep_river_arr.astype('uint8')
    deep_river_arr = cv2.erode(deep_river_img, kernel, iterations=1)
    deep_river_arr = np.where(deep_river_arr > 125, 1, 0)

    # Colors array and converts to pygame surface
    continent_grad = add_color(world_noise)
    # Turns map into a scaled pygame surface to draw on the screen.
    new_continent = np.uint8(continent_grad)
    pg.surfarray.blit_array(map_surface, new_continent)
    temp_surf = pg.transform.rotate(map_surface, -90)
    temp_surf2 = pg.transform.flip(temp_surf, True, False)
    scale_map_surface = pg.transform.scale(temp_surf2, (500, 500))
    update_button.show()
    tmx_button.show()
    print("Done")

# Creates TMX file based off of noise+mask
def make_tmx():
    global world_noise, random_plant_noise, rivers, river_edges, deep_river_arr
    print("Saving tmx file...")
    # The wave_base_arr is used to create an underbase for wave placement that doesn't include all the noise. For smoother wave patterns.
    temp_array = np.array(world_noise)
    wave_base_arr = temp_array * 255
    wave_base_img = wave_base_arr.astype('uint8')
    #cv2.imshow('image', wave_base_img)
    #cv2.waitKey(0)
    wave_base_img = cv2.medianBlur(wave_base_img, 5)
    wave_base_arr = wave_base_img / 255
    world_noise = np.where(ocean_regions == 1, wave_base_arr, world_noise) # Combines the medianBlur filtered ocean with the base layer.
    #world_noise = np.where(world_noise < 0, 0, world_noise) # Gets rid of negatives.

    max_num = 0
    min_num = 20
    #Makes base layer
    base_layer_vals = []
    water_layer_vals = []
    for i in range(MAP_SIZE):
        new_base_row = []
        new_water_row = []
        for j in range(MAP_SIZE):
            x = world_noise[i][j]
            newx = int(x * (KINDSOFTILES-1)) + 1
            if newx > KINDSOFTILES: # Prevents values from going out of range.
                newx = KINDSOFTILES
            if swamps_arr[i][j] == 1: # Swiches tiles to swamp tiles for swamp areas.
                newx = newx + KINDSOFTILES #Switches to swamp tileset columns
                if newx > TILESET_COLUMNS: # Prevents out of range swamp values.
                    newx = TILESET_COLUMNS
            if newx > max_num:
                max_num = newx
            if newx < min_num:
                min_num = newx
            # Adds water tiles for oceans lakes and swamps.
            water = 0
            if newx == 26:
                water = SWAMP_SHALLOWS
            elif KINDSOFTILES < newx < 26:
                water = SWAMP_WATER
            elif (newx == 5):
                if (ocean_regions[i][j] == 1):
                    water = SHALLOW_TILE
                else:
                    water = LAKE_SHALLOWS
            elif (newx < 5):
                if (ocean_regions[i][j] == 1):
                    water = WATER_TILE
                else:
                    water = LAKE_WATER
            new_water_row.append(water)
            new_base_row.append(newx)
        base_layer_vals.append(new_base_row)
        water_layer_vals.append(new_water_row)

    # Makes layer for rounded edges
    print("Applying pattern matching algorithms to make rounded corners...")
    print("(This takes a few minutes)")
    corners_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), int)
    overlay_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), int) #used for corners that overlay water edges.
    overlay2_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), int) #used for corners that overlay water edges.
    wave_val_arr = np.zeros((MAP_SIZE + 2, MAP_SIZE + 2), int)
    # Creates a temporary array with a padding of 1 to search for replace pattern
    arr1 = np.pad(base_layer_vals, pad_width=1, mode='constant', constant_values=0)
    # I know I screwed up and my x's and y's are switched in this loop.... But it works.
    for x in range(1, MAP_SIZE + 1):      # In this code I'm finding tile patterns by counting tiles of neighboring types in 3x3 and 2x2 groups instead of checking each tile in the group.
        print("Column " + str(x) + " out of " + str(MAP_SIZE) + ".")
        for y in range(1, MAP_SIZE + 1):
            # Scans map 4 blocks at a time to find the corner pattern.
            find_list = [arr1[x][y], arr1[x+1][y], arr1[x][y+1], arr1[x+1][y+1]] # Makes a list of 4 tile block values. Checks for basic corners.
            find_list2 = find_list.copy()
            find_list2.extend([arr1[x - 1][y], arr1[x - 1][y -1], arr1[x][y - 1], arr1[x + 1][y - 1], arr1[x - 1][y + 1]])  # Makes a list of 9 tile block values. Advanced checks
            hv_list = [arr1[x - 1][y], arr1[x + 1][y], arr1[x][y + 1], arr1[x][y - 1]]
            lg_list = hv_list.copy()
            lg_list.append(arr1[x][y]) #perpendicular tiles + center tile
            diag_list = [arr1[x - 1][y - 1], arr1[x + 1][y - 1], arr1[x - 1][y + 1], arr1[x + 1][y + 1]]
            ns_diag_list = [arr1[x - 1][y - 1], arr1[x + 1][y + 1]]
            ps_diag_list = [arr1[x + 1][y - 1], arr1[x - 1][y + 1]]

            for tile in range(TILESET_COLUMNS, 0, -1):
                num_tiles = find_list.count(tile) # Counts how many times each tile appears in a 2x2 block. If it's 1 then it's a corner.
                num_tiles2 = find_list2.count(tile) # Counts how many times teach tile type appears in a  3x3 block.
                num_under_tiles = find_list.count(tile - 1)  # Counts the number of the same tile that should be directly below the target tile there are.
                num_under_tiles2 = find_list2.count(tile - 1)
                num_hv_tiles = hv_list.count(tile)
                num_hv_under_tiles = hv_list.count(tile - 1)
                num_lg_tiles = lg_list.count(tile)
                num_lg_under_tiles = lg_list.count(tile - 1)
                num_diag_tiles = diag_list.count(tile)
                num_diag_under_tiles = diag_list.count(tile - 1)
                num_ns_diag_tiles = ns_diag_list.count(tile)
                num_ps_diag_tiles = ps_diag_list.count(tile)

                # rounds out positive corners and adds corner waves.
                if (num_tiles == 1) and (num_under_tiles == 3):
                    if arr1[x][y] == tile:
                        corners_layer_vals[x - 1][y - 1] = ul_corner(tile)
                        if (tile < 7) and (ocean_regions[x - 1][y - 1] == 1):
                            wave_val_arr[x][y] = WAVE_TILES[0]
                    elif arr1[x+1][y] == tile:
                        corners_layer_vals[x][y - 1] = ur_corner(tile)
                        if (tile < 7) and (ocean_regions[x][y - 1] == 1):
                            wave_val_arr[x + 1][y] = WAVE_TILES[5]
                    elif arr1[x][y+1] == tile:
                        corners_layer_vals[x - 1][y] = ll_corner(tile)
                        if (tile < 7) and (ocean_regions[x - 1][y] == 1):
                            wave_val_arr[x][y + 1] = WAVE_TILES[2]
                    elif arr1[x+1][y+1] == tile:
                        corners_layer_vals[x][y] = lr_corner(tile)
                        if (tile < 7) and (ocean_regions[x][y] == 1):
                            wave_val_arr[x + 1][y + 1] = WAVE_TILES[7]

                # Adds perpendicular waves.
                if (tile < 6) and (ocean_regions[x - 1][y - 1] == 1) and (arr1[x][y] == tile) and (wave_val_arr[x][y] == 0):
                    if (arr1[x + 1][y] == tile + 1) and (wave_val_arr[x + 1][y] == 0):
                        wave_val_arr[x][y] = WAVE_TILES[6]
                    if (arr1[x - 1][y] == tile + 1) and (wave_val_arr[x - 1][y] == 0):
                        wave_val_arr[x][y] = WAVE_TILES[1]
                    if (arr1[x][y + 1] == tile + 1) and (wave_val_arr[x][y + 1] == 0):
                        wave_val_arr[x][y] = WAVE_TILES[4]
                    if (arr1[x][y - 1] == tile + 1) and (wave_val_arr[x][y - 1] == 0):
                        wave_val_arr[x][y] = WAVE_TILES[3]

                if (num_hv_tiles == 4) and (tile - 1 == arr1[x][y]): # Fixes ugly single holes
                    if tile in [6, 27]:
                        overlay2_layer_vals[x -1][y - 1] = tile + (4 * TILESET_COLUMNS) # Sand water pools in water layer instead of corners layer.
                        break
                    else:
                        corners_layer_vals[x -1][y - 1] = tile + (4 * TILESET_COLUMNS)
                        break
                elif (num_hv_under_tiles == 4) and (tile == arr1[x][y]): # Fixes ugly single tile
                    corners_layer_vals[x -1][y - 1] = tile + (5 * TILESET_COLUMNS)
                    if (wave_val_arr[x][y] != 0): #Gets rid of waves that shouldn't be here.
                        wave_val_arr[x][y] = 0
                    if num_diag_tiles == 1:
                        if diag_list[0] == tile: #-- arr1[x - 1][y - 1]
                            overlay_layer_vals[x - 1][y - 1] = roundedpoint0(tile)
                            break
                        elif diag_list[1] == tile: #+-  arr1[x + 1][y - 1]
                            overlay_layer_vals[x - 1][y - 1] = roundedpoint1(tile)
                            break
                        elif diag_list[2] == tile: #-+  arr1[x - 1][y + 1]
                            overlay_layer_vals[x - 1][y - 1] = roundedpoint2(tile)
                            break
                        elif diag_list[3] == tile: #++  arr1[x + 1][y + 1]
                            overlay_layer_vals[x - 1][y - 1] = roundedpoint3(tile)
                            break
                    elif num_diag_tiles == 2: # Fixes single tiles with diagonal neighbors
                        if diag_list[0] + diag_list[1] == tile * 2:
                            corners_layer_vals[x - 1][y - 1] = pu_corner(tile)
                            break
                        elif diag_list[0] + diag_list[2] == tile * 2:
                            corners_layer_vals[x - 1][y - 1] = pl_corner(tile)
                            break
                        elif diag_list[0] + diag_list[3] == tile * 2:
                            if tile not in [6, 27]:
                                overlay_layer_vals[x - 1][y - 1] = roundedpoint0(tile)
                                overlay2_layer_vals[x - 1][y - 1] = roundedpoint3(tile)
                            break
                        elif diag_list[1] + diag_list[2] == tile * 2:
                            if tile not in [6, 27]:
                                overlay_layer_vals[x - 1][y - 1] = roundedpoint1(tile)
                                overlay2_layer_vals[x - 1][y - 1] = roundedpoint2(tile)
                            break
                        elif diag_list[1] + diag_list[3] == tile * 2:
                            corners_layer_vals[x - 1][y - 1] = pr_corner(tile)
                            break
                        elif diag_list[2] + diag_list[3] == tile * 2:
                            corners_layer_vals[x - 1][y - 1] = pd_corner(tile)
                            break

                    elif num_diag_tiles == 3:
                        if diag_list[0] + diag_list[1] + diag_list[2] == tile * 3:
                            corners_layer_vals[x - 1][y - 1] = ul_corner(tile)
                            break
                        elif diag_list[0] + diag_list[1] + diag_list[3] == tile * 3:
                            corners_layer_vals[x - 1][y - 1] = ur_corner(tile)
                            break
                        elif diag_list[0] + diag_list[2] + diag_list[3] == tile * 3:
                            corners_layer_vals[x - 1][y - 1] = ll_corner(tile)
                            break
                        elif diag_list[1] + diag_list[2] + diag_list[3] == tile * 3:
                            corners_layer_vals[x - 1][y - 1] = lr_corner(tile)
                            break
                    break

                elif (num_hv_tiles == 3) and (num_lg_under_tiles == 2): # rounds corners of 1 by x gaps
                    if arr1[x-1][y] == tile - 1:
                        if tile in [6, 27]:
                            overlay2_layer_vals[x - 1][y - 1] = td_corner(tile)
                            break
                        else:
                            corners_layer_vals[x - 1][y - 1] = td_corner(tile)
                            break
                    elif arr1[x+1][y] == tile - 1:
                        if tile in [6, 27]:
                            overlay2_layer_vals[x - 1][y - 1] = tu_corner(tile)
                            break
                        else:
                            corners_layer_vals[x - 1][y - 1] = tu_corner(tile)
                            break
                    elif arr1[x][y+1] == tile - 1:
                        if tile in [6, 27]:
                            overlay2_layer_vals[x - 1][y - 1] = tl_corner(tile)
                            break
                        else:
                            corners_layer_vals[x - 1][y - 1] = tl_corner(tile)
                            break
                    elif arr1[x][y-1] == tile - 1: #left
                        if tile in [6, 27]:
                            overlay2_layer_vals[x - 1][y - 1] = tr_corner(tile)
                            break
                        else:
                            corners_layer_vals[x - 1][y - 1] = tr_corner(tile)
                            break

                elif (num_hv_under_tiles == 3) and (num_lg_tiles == 2) and (num_ps_diag_tiles < 2) and (num_ns_diag_tiles < 2):  # Finds and places the rounded peninsula shaped tiles.
                    if (arr1[x-1][y] == tile) and (arr1[x+1][y - 1] != tile) and (arr1[x+1][y + 1] != tile):
                        corners_layer_vals[x - 1][y - 1] = pl_corner(tile)
                        if ocean_regions[x-1][y-1]:
                            wave_val_arr[x][y] = WAVE_TILES[1]
                        break
                    elif arr1[x+1][y] == tile and (arr1[x-1][y - 1] != tile) and (arr1[x-1][y + 1] != tile):
                        corners_layer_vals[x - 1][y - 1] = pr_corner(tile)
                        if ocean_regions[x - 1][y - 1]:
                            wave_val_arr[x][y] = WAVE_TILES[6]
                        break
                    elif arr1[x][y+1] == tile and (arr1[x-1][y - 1] != tile) and (arr1[x+1][y - 1] != tile):
                        corners_layer_vals[x - 1][y - 1] = pd_corner(tile)
                        if ocean_regions[x - 1][y - 1]:
                            wave_val_arr[x][y] = WAVE_TILES[4]
                        break
                    elif arr1[x][y-1] == tile and (arr1[x-1][y + 1] != tile) and (arr1[x+1][y + 1] != tile): #left
                        corners_layer_vals[x - 1][y - 1] = pu_corner(tile)
                        if ocean_regions[x - 1][y - 1]:
                            wave_val_arr[x][y] = WAVE_TILES[3]
                        break

                elif (num_hv_tiles == 2) and (arr1[x][y] == tile - 1): # Rounds out negative corners
                    if arr1[x - 1][y] + arr1[x][y - 1] == tile*2:
                        if tile in [6, 27]:
                            overlay2_layer_vals[x - 1][y - 1] = nul_corner(tile) # Beach corners
                        else:
                            corners_layer_vals[x - 1][y - 1] = nul_corner(tile)
                        break
                    elif arr1[x + 1][y] + arr1[x][y - 1] == tile*2:
                        if tile in [6, 27]:
                            overlay2_layer_vals[x - 1][y - 1] = nur_corner(tile) # Beach corners
                        else:
                            corners_layer_vals[x -1][y - 1] = nur_corner(tile)
                        break
                    elif arr1[x - 1][y] + arr1[x][y + 1] == tile*2:
                        if tile in [6, 27]:
                            overlay2_layer_vals[x - 1][y - 1] = nll_corner(tile) # Beach corners
                        else:
                            corners_layer_vals[x - 1][y - 1] = nll_corner(tile)
                        break
                    elif arr1[x + 1][y] + arr1[x][y + 1] == tile*2:
                        if tile in [6, 27]:
                            overlay2_layer_vals[x - 1][y - 1] = nlr_corner(tile) # Beach corners
                        else:
                            corners_layer_vals[x - 1][y - 1] = nlr_corner(tile)
                        break

    # This secondary loop finishes wave layer by removing anomalous waves and filling in gaps.
    # Creates a temporary array with a padding of 1 to search for replace pattern
    print("Adding waves...")
    wave2_val_arr = np.zeros((MAP_SIZE + 2, MAP_SIZE + 2), int)
    for y in range(1, MAP_SIZE + 1):
        for x in range(1, MAP_SIZE + 1):
            # Scans map 4 blocks at a time to find the corner pattern.
            find_list = [wave_val_arr[y][x], wave_val_arr[y+1][x], wave_val_arr[y][x+1], wave_val_arr[y+1][x+1]] # Makes a list of 4 tile block values. Checks for basic corners.
            find_list2 = find_list.copy()
            find_list2.extend([wave_val_arr[y - 1][x], wave_val_arr[y - 1][x -1], wave_val_arr[y][x - 1], wave_val_arr[y + 1][x - 1], wave_val_arr[y - 1][x + 1]])  # Makes a list of 9 tile block values. Advanced checks
            hv_list = [wave_val_arr[y - 1][x], wave_val_arr[y + 1][x], wave_val_arr[y][x + 1], wave_val_arr[y][x - 1]]
            diag_list = [wave_val_arr[y - 1][x - 1], wave_val_arr[y + 1][x - 1], wave_val_arr[y - 1][x + 1], wave_val_arr[y + 1][x + 1]]
            num_waves2 = 9 - find_list2.count(0)
            num_diag = 4 - diag_list.count(0)
            num_hv = 4 - hv_list.count(0)


            if wave_val_arr[y][x] == 0: # Only modifies empty cells.
                if num_waves2 == 2:
                    # Fills in gaps between parallel waves
                    if wave_val_arr[y - 1][x] and wave_val_arr[y + 1][x]:
                        if arr1[y][x] == arr1[y][x + 1] - 1:
                            wave2_val_arr[y][x] = WAVE_TILES[4]
                        elif arr1[y][x] == arr1[y][x - 1] - 1:
                            wave2_val_arr[y][x] = WAVE_TILES[3]
                        else: # Write the code for the waves that cross like base layer.
                            pass
                    elif wave_val_arr[y][x - 1] and wave_val_arr[y][x + 1]:
                        if arr1[y][x] == arr1[y + 1][x] - 1:
                            wave2_val_arr[y][x] = WAVE_TILES[6]
                        elif arr1[y][x] == arr1[y - 1][x] - 1:
                            wave2_val_arr[y][x] = WAVE_TILES[1]
                        else:
                            pass

                    #Fills in missing diagonal between two diagonals.
                    elif wave_val_arr[y + 1][x - 1] and wave_val_arr[y - 1][x + 1]:
                        if wave_val_arr[y + 1][x - 1] == WAVE_TILES[7]:
                            wave2_val_arr[y][x] = WAVE_TILES[7]
                        elif wave_val_arr[y + 1][x - 1] == WAVE_TILES[0]:
                            wave2_val_arr[y][x] = WAVE_TILES[0]
                    elif wave_val_arr[y - 1][x - 1] and wave_val_arr[y + 1][x + 1]:
                        if wave_val_arr[y - 1][x - 1] == WAVE_TILES[2]:
                            wave2_val_arr[y][x] = WAVE_TILES[2]
                        elif wave_val_arr[y - 1][x - 1] == WAVE_TILES[5]:
                            wave2_val_arr[y][x] = WAVE_TILES[5]

                    elif (num_diag == 1) and (num_hv == 1):
                        if arr1[y][x] == arr1[y + 1][x] - 1:
                            if wave_val_arr[y + 1][x] == 0:
                                wave2_val_arr[y][x] = WAVE_TILES[6]
                        if arr1[y][x] == arr1[y - 1][x] - 1:
                            if wave_val_arr[y - 1][x] == 0:
                                wave2_val_arr[y][x] = WAVE_TILES[1]
                        if arr1[y][x] == arr1[y][x + 1] - 1:
                            if wave_val_arr[y][x + 1] == 0:
                                if wave_val_arr[y + 1][x - 1] == WAVE_TILES[7]:
                                    wave2_val_arr[y][x] = WAVE_TILES[7]
                                elif wave_val_arr[y - 1][x - 1] == WAVE_TILES[2]:
                                    wave2_val_arr[y][x] = WAVE_TILES[2]
                                elif wave_val_arr[y][x - 1] == WAVE_TILES[6]:
                                    wave2_val_arr[y][x] = WAVE_TILES[7]
                                elif wave_val_arr[y][x - 1] == WAVE_TILES[1]:
                                    wave2_val_arr[y][x] = WAVE_TILES[2]
                                else:
                                    wave2_val_arr[y][x] = WAVE_TILES[4]
                        if arr1[y][x] == arr1[y][x - 1] - 1:
                            if wave_val_arr[y][x - 1] == 0:
                                if wave_val_arr[y + 1][x + 1] == WAVE_TILES[5]:
                                    wave2_val_arr[y][x] = WAVE_TILES[5]
                                elif wave_val_arr[y - 1][x + 1] == WAVE_TILES[0]:
                                    wave2_val_arr[y][x] = WAVE_TILES[0]
                                elif wave_val_arr[y][x + 1] == WAVE_TILES[6]:
                                    wave2_val_arr[y][x] = WAVE_TILES[5]
                                elif wave_val_arr[y][x + 1] == WAVE_TILES[1]:
                                    wave2_val_arr[y][x] = WAVE_TILES[0]
                                else:
                                    wave2_val_arr[y][x] = WAVE_TILES[3]
    wave_val_arr = np.where(wave2_val_arr == 0, wave_val_arr, wave2_val_arr)

    wave0_val_arr = np.zeros((MAP_SIZE + 2, MAP_SIZE + 2), int)
    for y in range(1, MAP_SIZE + 1):
        for x in range(1, MAP_SIZE + 1):
            # Scans map 4 blocks at a time to find the corner pattern.
            find_list = [wave_val_arr[y][x], wave_val_arr[y+1][x], wave_val_arr[y][x+1], wave_val_arr[y+1][x+1]] # Makes a list of 4 tile block values. Checks for basic corners.
            find_list2 = find_list.copy()
            find_list2.extend([wave_val_arr[y - 1][x], wave_val_arr[y - 1][x -1], wave_val_arr[y][x - 1], wave_val_arr[y + 1][x - 1], wave_val_arr[y - 1][x + 1]])  # Makes a list of 9 tile block values. Advanced checks
            hv_list = [wave_val_arr[y - 1][x], wave_val_arr[y + 1][x], wave_val_arr[y][x + 1], wave_val_arr[y][x - 1]]
            diag_list = [wave_val_arr[y - 1][x - 1], wave_val_arr[y + 1][x - 1], wave_val_arr[y - 1][x + 1], wave_val_arr[y + 1][x + 1]]
            num_waves2 = 9 - find_list2.count(0)
            num_diag = 4 - diag_list.count(0)
            num_hv = 4 - hv_list.count(0)

            if wave_val_arr[y][x] == 0:  # Only modifies empty cells.
                if num_waves2 == 2:
                    # Fixes right angle wave corners.
                    if num_hv == 2:
                        if wave_val_arr[y][x - 1] in [WAVE_TILES[0], WAVE_TILES[1]]:
                            if wave_val_arr[y + 1][x] in [WAVE_TILES[4], WAVE_TILES[7]]:
                                wave0_val_arr[y][x] = WAVE_TILES[2]
                        elif wave_val_arr[y][x + 1] in [WAVE_TILES[1], WAVE_TILES[2]]:
                            if wave_val_arr[y + 1][x] in [WAVE_TILES[3], WAVE_TILES[5]]:
                                wave0_val_arr[y][x] = WAVE_TILES[0]
                        elif wave_val_arr[y][x + 1] in [WAVE_TILES[7], WAVE_TILES[6]]:
                            if wave_val_arr[y - 1][x] in [WAVE_TILES[3], WAVE_TILES[0]]:
                                wave0_val_arr[y][x] = WAVE_TILES[5]
                        elif wave_val_arr[y][x - 1] in [WAVE_TILES[5], WAVE_TILES[6]]:
                            if wave_val_arr[y - 1][x] in [WAVE_TILES[2], WAVE_TILES[4]]:
                                wave0_val_arr[y][x] = WAVE_TILES[7]
    wave_val_arr = np.where(wave0_val_arr == 0, wave_val_arr, wave0_val_arr)

    wave4_val_arr = np.zeros((MAP_SIZE + 4, MAP_SIZE + 4), int)
    temp_arr = np.pad(wave_val_arr, pad_width=1, mode='constant', constant_values=0) # Pads array by one more for 4 tile wide searches.
    for y in range(2, MAP_SIZE + 2):
        for x in range(2, MAP_SIZE + 2):
            if temp_arr[y][x] == 0:  # Only modifies empty cells.
                # Fills in two tile gaps
                if temp_arr[y][x + 1] == 0:
                    if temp_arr[y][x - 1] in [WAVE_TILES[0], WAVE_TILES[1]]:
                        if temp_arr[y - 1][x] + temp_arr[y - 1][x + 1] == 0:
                            if temp_arr[y][x + 2] in [WAVE_TILES[1], WAVE_TILES[2]]:
                                wave4_val_arr[y][x] = WAVE_TILES[1]
                                wave4_val_arr[y][x + 1] = WAVE_TILES[1]
                    if temp_arr[y][x - 1] in [WAVE_TILES[5], WAVE_TILES[6]]:
                        if temp_arr[y + 1][x] + temp_arr[y + 1][x + 1] == 0:
                            if temp_arr[y][x + 2] in [WAVE_TILES[6], WAVE_TILES[7]]:
                                wave4_val_arr[y][x] = WAVE_TILES[6]
                                wave4_val_arr[y][x + 1] = WAVE_TILES[6]
                if temp_arr[y + 1][x] == 0:
                    if temp_arr[y - 1][x] in [WAVE_TILES[0], WAVE_TILES[3]]:
                        if temp_arr[y][x - 1] + temp_arr[y + 1][x - 1] == 0:
                            if temp_arr[y + 2][x] in [WAVE_TILES[3], WAVE_TILES[5]]:
                                wave4_val_arr[y][x] = WAVE_TILES[3]
                                wave4_val_arr[y + 1][x] = WAVE_TILES[3]
                    if temp_arr[y - 1][x] in [WAVE_TILES[2], WAVE_TILES[4]]:
                        if temp_arr[y][x + 1] + temp_arr[y + 1][x + 1] == 0:
                            if temp_arr[y + 2][x] in [WAVE_TILES[4], WAVE_TILES[7]]:
                                wave4_val_arr[y][x] = WAVE_TILES[4]
                                wave4_val_arr[y + 1][x] = WAVE_TILES[4]
    wave4_val_arr = wave4_val_arr[1:-1, 1:-1]  # Unpads the array by 1.
    wave_val_arr = np.where(wave4_val_arr == 0, wave_val_arr, wave4_val_arr)

    wave3_val_arr = np.zeros((MAP_SIZE + 2, MAP_SIZE + 2), int)
    for y in range(1, MAP_SIZE + 1):
        for x in range(1, MAP_SIZE + 1):
            # Adds tiny wave corners
            if wave_val_arr[y][x] == 0:
                # pos slope slanting waves
                if wave_val_arr[y][x + 1] and wave_val_arr[y + 1][x]:
                    if wave_val_arr[y + 1][x] in [WAVE_TILES[0], WAVE_TILES[1]]:
                        wave3_val_arr[y][x] = WAVE_CORNER_TILES[0]
                        wave3_val_arr[y + 1][x + 1] = WAVE_CORNER_TILES2[0]
                    elif wave_val_arr[y][x + 1] in [WAVE_TILES[7], WAVE_TILES[6]]:
                        wave3_val_arr[y][x] = WAVE_CORNER_TILES2[3]
                        wave3_val_arr[y + 1][x + 1] = WAVE_CORNER_TILES[3]
                # neg slope slanting waves
                if wave_val_arr[y][x - 1] and wave_val_arr[y + 1][x]:
                    if wave_val_arr[y][x - 1] in [WAVE_TILES[5], WAVE_TILES[6]]:
                        wave3_val_arr[y][x] = WAVE_CORNER_TILES2[2]
                        wave3_val_arr[y + 1][x - 1] = WAVE_CORNER_TILES[2]
                    elif wave_val_arr[y + 1][x] in [WAVE_TILES[2], WAVE_TILES[1]]:
                        wave3_val_arr[y][x] = WAVE_CORNER_TILES[1]
                        wave3_val_arr[y + 1][x - 1] = WAVE_CORNER_TILES2[1]
                # side by side switching waves.
                if wave_val_arr[y + 1][x] and wave_val_arr[y + 1][x + 1]:
                    if (wave_val_arr[y + 1][x] == WAVE_TILES[7]) and (wave_val_arr[y + 1][x + 1] == WAVE_TILES[5]):
                        wave3_val_arr[y][x] = WAVE_CORNER_TILES2[3]
                        wave3_val_arr[y][x + 1] = WAVE_CORNER_TILES2[2]
                    elif (wave_val_arr[y + 1][x] == WAVE_TILES[2]) and (wave_val_arr[y + 1][x + 1] == WAVE_TILES[0]):
                        try:
                            wave3_val_arr[y + 2][x] = WAVE_CORNER_TILES2[1]
                            wave3_val_arr[y + 2][x + 1] = WAVE_CORNER_TILES2[0]
                        except:
                            pass
                if wave_val_arr[y][x + 1] and wave_val_arr[y + 1][x + 1]:
                    if (wave_val_arr[y][x + 1] == WAVE_TILES[7]) and (wave_val_arr[y + 1][x + 1] == WAVE_TILES[2]):
                        wave3_val_arr[y][x] = WAVE_CORNER_TILES2[3]
                        wave3_val_arr[y + 1][x] = WAVE_CORNER_TILES2[1]
                    elif (wave_val_arr[y][x + 1] == WAVE_TILES[5]) and (wave_val_arr[y + 1][x + 1] == WAVE_TILES[0]):
                        try:
                            wave3_val_arr[y][x + 2] = WAVE_CORNER_TILES2[2]
                            wave3_val_arr[y + 1][x + 2] = WAVE_CORNER_TILES2[0]
                        except:
                            pass
    wave_val_arr = np.where(wave3_val_arr == 0, wave_val_arr, wave3_val_arr)
    waves_layer_vals = wave_val_arr[1:-1, 1:-1] #Unpads the array.
    
    # Alters river tiles to look better by changing some tiles to non directional lake water. Refines river rock edges.
    arr1 = np.pad(rivers, pad_width=1, mode='constant', constant_values=0)
    # I know I screwed up and my x's and y's are switched in this loop.... But it works.
    for y in range(1, MAP_SIZE + 1):  
        for x in range(1, MAP_SIZE + 1):
            # Scans map 4 blocks at a time to find the corner pattern.
            find_list = [arr1[y][x], arr1[y+1][x], arr1[y][x+1], arr1[y+1][x+1]] # Makes a list of 4 tile block values. Checks for basic corners.
            if (arr1[y][x] == 0) and (arr1[y][x+1] > 0) and (river_edges[y - 1][x - 1] == 0):
                river_edges[y - 1][x - 1] = hflip(RIVER_ROCKS[3])
            if (arr1[y][x] > 0) and (arr1[y][x+1] == 0) and (river_edges[y - 1][x] == 0):
                river_edges[y - 1][x] = RIVER_ROCKS[3]
            if (arr1[y][x] == 0) and (arr1[y + 1][x] > 0) and (river_edges[y - 1][x - 1] == 0):
                river_edges[y - 1][x - 1] = vdflip(RIVER_ROCKS[3])
            if (arr1[y][x] > 0) and (arr1[y + 1][x] == 0) and (river_edges[y][x - 1] == 0):
                river_edges[y][x - 1] = dflip(RIVER_ROCKS[3])

            if 0 not in find_list:
                if not all(element == find_list[0] for element in find_list): # Checks to see if all river elements are the same if not changes them into lake water.
                    rivers[y - 1][x - 1] = rivers[y][x - 1] = rivers[y - 1][x] = rivers[y][x] = LAKE_SHALLOWS
            if find_list.count(0) == 1:
                if rivers[y - 1][x - 1] > 0:
                    rivers[y - 1][x - 1] = LAKE_SHALLOWS
                elif 0 in [river_edges[y - 1][x - 1], river_edges2[y - 1][x - 1]]:
                        river_edges[y - 1][x - 1] = hflip(RIVER_ROCKS[3])
                        river_edges2[y - 1][x - 1] = vdflip(RIVER_ROCKS[3])
                if rivers[y][x - 1] > 0:
                    rivers[y][x - 1] = LAKE_SHALLOWS
                elif 0 in [river_edges[y][x - 1], river_edges2[y][x - 1]]:
                    river_edges[y][x - 1] = hflip(RIVER_ROCKS[3])
                    river_edges2[y][x - 1] = dflip(RIVER_ROCKS[3])
                if rivers[y - 1][x] > 0:
                    rivers[y - 1][x] = LAKE_SHALLOWS
                elif 0 in [river_edges[y - 1][x], river_edges2[y - 1][x]]:
                    river_edges[y - 1][x] = RIVER_ROCKS[3]
                    river_edges2[y - 1][x] = vdflip(RIVER_ROCKS[3])
                if rivers[y][x] > 0:
                    rivers[y][x] = LAKE_SHALLOWS
                elif 0 in [river_edges[y][x], river_edges2[y][x]]:
                    river_edges[y][x] = RIVER_ROCKS[3]
                    river_edges2[y][x] = dflip(RIVER_ROCKS[3])
            if find_list.count(0) == 3: # Adds river rock edges.
                if rivers[y - 1][x - 1] > 0:
                    river_edges[y][x] = RIVER_ROCKS[4]
                if rivers[y][x - 1] > 0:
                    river_edges[y - 1][x] = vflip(RIVER_ROCKS[4])
                if rivers[y - 1][x] > 0:
                    river_edges[y][x - 1] = hflip(RIVER_ROCKS[4])
                if rivers[y][x] > 0:
                    river_edges[y - 1][x - 1] = hvflip(RIVER_ROCKS[4])


    plant_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), int)
    tree_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), int)
    # Places vegetation
    print("Addinig plants...")
    for y in range(0, MAP_SIZE):
        for x in range(0, MAP_SIZE):
            if rivers[y][x] == 0:
                if (ocean_regions[y][x] == 0) and (y < MAP_SIZE - TREE_DIMENTIONS) and (x < MAP_SIZE - TREE_DIMENTIONS): # Keeps it for going out of range.
                    if random.randrange(0, 10): # Only does tree checks every 10th tile on average.
                        tree_scope = [tree_layer_vals[y][x], tree_layer_vals[y][x+1], tree_layer_vals[y][x+2], tree_layer_vals[y][x+3], tree_layer_vals[y][x+4], tree_layer_vals[y][x+5], tree_layer_vals[y][x+6],
                                      tree_layer_vals[y+1][x], tree_layer_vals[y+1][x+1], tree_layer_vals[y+1][x+2], tree_layer_vals[y+1][x+3], tree_layer_vals[y+1][x+4], tree_layer_vals[y+1][x+5], tree_layer_vals[y+1][x+6],
                                      tree_layer_vals[y +2][x], tree_layer_vals[y +2][x+1], tree_layer_vals[y +2][x+2], tree_layer_vals[y +2][x+3], tree_layer_vals[y +2][x+4], tree_layer_vals[y +2][x+5], tree_layer_vals[y +2][x+6],
                                      tree_layer_vals[y +3][x], tree_layer_vals[y +3][x+1], tree_layer_vals[y +3][x+2], tree_layer_vals[y +3][x+3], tree_layer_vals[y +3][x+4], tree_layer_vals[y +3][x+5], tree_layer_vals[y +3][x+6],
                                      tree_layer_vals[y +4][x], tree_layer_vals[y +4][x+1], tree_layer_vals[y +4][x+2], tree_layer_vals[y +4][x+3], tree_layer_vals[y +4][x+4], tree_layer_vals[y +4][x+5], tree_layer_vals[y +4][x+6],
                                      tree_layer_vals[y +5][x], tree_layer_vals[y +5][x+1], tree_layer_vals[y +5][x+2], tree_layer_vals[y +5][x+3], tree_layer_vals[y +5][x+4], tree_layer_vals[y +5][x+5], tree_layer_vals[y +5][x+6],
                                      tree_layer_vals[y +6][x], tree_layer_vals[y +6][x+1], tree_layer_vals[y +6][x+2], tree_layer_vals[y +6][x+3], tree_layer_vals[y +6][x+4], tree_layer_vals[y +6][x+5], tree_layer_vals[y +6][x+6]]
                        # Trees
                        place_tree = True
                        if beach_regions[y][x] and (6 < base_layer_vals[y][x] < 11):
                            tree = MAIN_TREE_TILES[2]
                            prob = PALM_PROB
                        elif (not beach_regions[y][x]) and base_layer_vals[y][x] in [8, 9]:
                            tree = MAIN_TREE_TILES[1]
                            prob = GREENTREE_PROB
                        elif (not beach_regions[y][x]) and base_layer_vals[y][x] in [10, 11]:
                            if random.randrange(0, 3) == 1:
                                tree = MAIN_TREE_TILES[3]
                                prob = PINE_PROB
                            elif random.randrange(0, 10) == 1:
                                tree = MAIN_TREE_TILES[0]
                                prob = DEADTREE_PROB
                            else:
                                tree = MAIN_TREE_TILES[1]
                                prob = GREENTREE_PROB
                        elif (not beach_regions[y][x]) and (11 < base_layer_vals[y][x] < 14):
                            if random.randrange(0, 10) == 1:
                                tree = MAIN_TREE_TILES[0]
                                prob = DEADTREE_PROB
                            else:
                                tree = MAIN_TREE_TILES[3]
                                prob = PINE_PROB * 2
                        elif (not beach_regions[y][x]) and (14 < base_layer_vals[y][x] < 17):
                            if random.randrange(0, 10) == 1:
                                tree = MAIN_TREE_TILES[0]
                                prob = DEADTREE_PROB * 4
                            else:
                                tree = MAIN_TREE_TILES[3]
                                prob = PINE_PROB * 4
                        else:
                            place_tree = False
                        if place_tree and (random.randrange(0, prob * 10) == 1):
                            for tile in tree_scope:
                                if tile in ALL_TREE_TILES: # checks to see if there are already tree tiles before placing a tree.
                                    place_tree = False
                                    continue
                            if place_tree:
                                tile = tree
                                for i in range(0, TREE_DIMENTIONS):
                                    for j in range(0, TREE_DIMENTIONS):
                                        tree_layer_vals[y + i][x + j] = tile
                                        tile += 1
                                    tile += (TILESET_COLUMNS - TREE_DIMENTIONS)

                # Plants
                if base_layer_vals[y][x] == 8: # If grassland.
                    if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                        for plant, prob in GRASSLAND_PLANTS.items():
                            if random.randrange(0, GPLANT_FACT*prob) == 1:
                                plant_layer_vals[y][x] = plant
                elif (water_layer_vals[y][x] == WATER_TILE) and (ocean_regions[y][x]): # If ocean.
                    if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                        if base_layer_vals[y][x] == 4:
                            pfact = OPLANT_FACT * 10
                        elif base_layer_vals[y][x] == 3:
                            pfact = OPLANT_FACT * 3
                        elif base_layer_vals[y][x] == 2:
                            pfact = OPLANT_FACT
                        elif base_layer_vals[y][x] == 1:
                            pfact = OPLANT_FACT * 100
                        else:
                            continue
                        for plant, prob in OCEAN_PLANTS.items():
                            if random.randrange(0, pfact*prob) == 1:
                                corners_layer_vals[y][x] = plant # Puts them under the water.
                elif base_layer_vals[y][x] == 10:  # Forest
                    if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                        for plant, prob in FOREST_PLANTS.items():
                            if random.randrange(0, FPLANT_FACT*prob) == 1:
                                plant_layer_vals[y][x] = plant
                elif water_layer_vals[y][x] in [SWAMP_SHALLOWS]:  # Swamp water
                    if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                        for plant, prob in SWAMPW_PLANTS.items():
                            if random.randrange(0, SPLANTW_FACT*prob) == 1:
                                plant_layer_vals[y][x] = plant
                elif base_layer_vals[y][x] in [26, 27]:  # swamp (not in water)
                    if corners_layer_vals[y][x] == 0:
                        for plant, prob in SWAMPD_PLANTS.items():
                            if random.randrange(0, SPLANTD_FACT * prob) == 1:
                                plant_layer_vals[y][x] = plant
                elif (base_layer_vals[y][x] == 6) and (not beach_regions[y][x]):  # desert
                    if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                        for plant, prob in DESERT_PLANTS.items():
                            if random.randrange(0, DPLANT_FACT*prob) == 1:
                                plant_layer_vals[y][x] = plant
                elif (base_layer_vals[y][x] == 7) and (not beach_regions[y][x]):  # dirt not near beaches
                    if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                        for plant, prob in WASTELAND_PLANTS.items():
                            if random.randrange(0, WPLANT_FACT*prob) == 1:
                                plant_layer_vals[y][x] = plant
                elif 10 < base_layer_vals[y][x] < 17: # If mountains
                    if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                        pfact = (base_layer_vals[y][x] - 10) * MPLANT_FACT
                        for plant, prob in MOUNTAIN_PLANTS.items():
                            if random.randrange(0, pfact*prob) == 1:
                                plant_layer_vals[y][x] = plant

    # Puts river rocks under rivers.
    rivers = np.where(deep_river_arr == 0, rivers, LAKE_WATER) # Creates deep water in ponds and lakes.
    river_edges = np.where(rivers == 0, river_edges, RIVER_ROCKS[0])
    overlay_layer_vals = np.where(river_edges == 0, overlay_layer_vals, river_edges)
    overlay2_layer_vals = np.where(river_edges2 == 0, overlay2_layer_vals, river_edges2)

    # Gets rid of flowing river edges along lakes, ponds and wide rivers
    arr1 = np.pad(rivers, pad_width=1, mode='constant', constant_values=0)
    # I know I screwed up and my x's and y's are switched in this loop.... But it works.
    for y in range(1, MAP_SIZE + 1):
        for x in range(1, MAP_SIZE + 1):
            # Scans map 4 blocks at a time to find the corner pattern.
            if arr1[y][x] in RIVER_TILES:
                if arr1[y+1][x] == LAKE_WATER:
                    rivers[y - 1][x - 1] = LAKE_SHALLOWS
                if arr1[y][x+1] == LAKE_WATER:
                    rivers[y - 1][x - 1] = LAKE_SHALLOWS
            if arr1[y+1][x] in RIVER_TILES:
                if arr1[y][x] == LAKE_WATER:
                    rivers[y][x - 1] = LAKE_SHALLOWS
            if arr1[y][x+1] in RIVER_TILES:
                if arr1[y][x] == LAKE_WATER:
                    rivers[y - 1][x] = LAKE_SHALLOWS

    # Writes base layer
    print("Writing base layer...")
    outfile = open("newautomap.tmx", "w")
    header = """<?xml version="1.0" encoding="UTF-8"?>
    <map version="1.4" tiledversion="1.4.1" orientation="orthogonal" renderorder="right-down" width="{mapw}" height="{mapw}" tilewidth="32" tileheight="32" infinite="0" nextlayerid="2" nextobjectid="1">
     <tileset firstgid="1" name="automaptiles" tilewidth="32" tileheight="32" tilecount="{tile_count}" columns="{tileset_columns}">
      <image source="automaptiles.png" width="448" height="32"/>
        <tile id="{watertile}">
          </tile>
     </tileset>
     <layer id="1" name="Base Layer" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE), watertile = str(LAKE_WATER), tile_count = str(TILE_COUNT), tileset_columns = str(TILESET_COLUMNS))
    outfile.write(header)
    outfile.write("\n")
    for i, row in enumerate(base_layer_vals):
        row_text = str(row)
        row_text = row_text.replace('[', '')
        row_text = row_text.replace(']', '')
        if i != MAP_SIZE - 1:
            row_text = row_text + ","
        outfile.write(row_text)
        outfile.write("\n")
    footer = """</data>
     </layer>"""
    outfile.write(footer)

    # Writes rounded corner layer to tmx file
    print("Writing rounded corner layer...")
    corners_list = np.ndarray.tolist(corners_layer_vals)
    next_layer_txt = """<layer id="2" name="Rounded Corners" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE))
    outfile.write(next_layer_txt)
    outfile.write("\n")
    for i, y in enumerate(corners_list):
        new_row_str = str(y)
        new_row_str = new_row_str.replace('[', '')
        new_row_str = new_row_str.replace(']', '')
        if i != MAP_SIZE - 1:
            new_row_str = new_row_str + ","
        outfile.write(new_row_str)
        outfile.write("\n")
    footer = """</data>
     </layer>"""
    outfile.write(footer)

    # Makes and writes overlay corners layer
    print("Writing overlay layer...")
    overlay_list = np.ndarray.tolist(overlay_layer_vals)
    next_layer_txt = """<layer id="3" name="Overlay Corners" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE))
    outfile.write(next_layer_txt)
    outfile.write("\n")
    for i, y in enumerate(overlay_list):
        new_row_str = str(y)
        new_row_str = new_row_str.replace('[', '')
        new_row_str = new_row_str.replace(']', '')
        if i != MAP_SIZE - 1:
            new_row_str = new_row_str + ","
        outfile.write(new_row_str)
        outfile.write("\n")
    footer = """</data>
     </layer>"""
    outfile.write(footer)

    # Makes and writes water layer
    print("Writing water layer...")
    next_layer_txt = """<layer id="4" name="Water" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE))
    outfile.write(next_layer_txt)
    outfile.write("\n")
    for i, y in enumerate(water_layer_vals):
        new_row_str = str(y)
        new_row_str = new_row_str.replace('[', '')
        new_row_str = new_row_str.replace(']', '')
        if i != MAP_SIZE - 1:
            new_row_str = new_row_str + ","
        outfile.write(new_row_str)
        outfile.write("\n")
    footer = """</data>
     </layer>"""
    outfile.write(footer)

    # Makes and writes overlay corners layer
    print("Writing overlay2 layer...")
    overlay2_list = np.ndarray.tolist(overlay2_layer_vals)
    next_layer_txt = """<layer id="5" name="Overlay2 Corners" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE))
    outfile.write(next_layer_txt)
    outfile.write("\n")
    for i, y in enumerate(overlay2_list):
        new_row_str = str(y)
        new_row_str = new_row_str.replace('[', '')
        new_row_str = new_row_str.replace(']', '')
        if i != MAP_SIZE - 1:
            new_row_str = new_row_str + ","
        outfile.write(new_row_str)
        outfile.write("\n")
    footer = """</data>
     </layer>"""
    outfile.write(footer)

    # Makes and writes rivers layer
    print("Writing rivers layer...")
    rivers_list = np.ndarray.tolist(rivers)
    next_layer_txt = """<layer id="6" name="Rivers" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE))
    outfile.write(next_layer_txt)
    outfile.write("\n")
    for i, y in enumerate(rivers_list):
        new_row_str = str(y)
        new_row_str = new_row_str.replace('[', '')
        new_row_str = new_row_str.replace(']', '')
        if i != MAP_SIZE - 1:
            new_row_str = new_row_str + ","
        outfile.write(new_row_str)
        outfile.write("\n")
    footer = """</data>
     </layer>"""
    outfile.write(footer)

    # Makes and writes waves and plants layer
    print("Writing waves layer...")
    waves_list = np.ndarray.tolist(waves_layer_vals)
    next_layer_txt = """<layer id="7" name="Waves" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE))
    outfile.write(next_layer_txt)
    outfile.write("\n")
    for i, y in enumerate(waves_list):
        new_row_str = str(y)
        new_row_str = new_row_str.replace('[', '')
        new_row_str = new_row_str.replace(']', '')
        if i != MAP_SIZE - 1:
            new_row_str = new_row_str + ","
        outfile.write(new_row_str)
        outfile.write("\n")
    footer = """</data>
     </layer>"""
    outfile.write(footer)

    # Makes and writes plants layer
    print("Writing plants layer...")
    plant_list = np.ndarray.tolist(plant_layer_vals)
    next_layer_txt = """<layer id="8" name="Plants" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE))
    outfile.write(next_layer_txt)
    outfile.write("\n")
    for i, y in enumerate(plant_list):
        new_row_str = str(y)
        new_row_str = new_row_str.replace('[', '')
        new_row_str = new_row_str.replace(']', '')
        if i != MAP_SIZE - 1:
            new_row_str = new_row_str + ","
        outfile.write(new_row_str)
        outfile.write("\n")
    footer = """</data>
     </layer>"""
    outfile.write(footer)

    # Makes and writes tree layer
    print("Writing tree layer...")
    tree_list = np.ndarray.tolist(tree_layer_vals)
    next_layer_txt = """<layer id="9" name="Trees" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE))
    outfile.write(next_layer_txt)
    outfile.write("\n")
    for i, y in enumerate(tree_list):
        new_row_str = str(y)
        new_row_str = new_row_str.replace('[', '')
        new_row_str = new_row_str.replace(']', '')
        if i != MAP_SIZE - 1:
            new_row_str = new_row_str + ","
        outfile.write(new_row_str)
        outfile.write("\n")
    footer = """</data>
     </layer>
    </map>"""
    outfile.write(footer)
    print("Max value: " + str(max_num))
    print("Min value: " + str(min_num))
    print("Finished.")
    outfile.close()

# UI elements
ui = ravenui.UI(screen)
mask_slider = ravenui.Slider(ui, "Ocean Depth", (10, 10), MASK_FACTOR, 0.4, 0, True)
elev_slider = ravenui.Slider(ui, "Land Density", (112, 10), ELEVATION_FACTOR, 0.5, -0.9, True)
sealvl_slider = ravenui.Slider(ui, "Elevation", (214, 10), SEALVL_FACTOR, 1.3, 0.0001, True)
scale_slider = ravenui.Slider(ui, "Scale", (316, 10), 100, 200, 5, True)
octaves_slider = ravenui.Slider(ui, "Octaves", (418, 10), 6, 10, 1)
persistence_slider = ravenui.Slider(ui, "Persistence", (520, 10), 0.5, 1, 0.1, True)
lacunarity_slider = ravenui.Slider(ui, "Lacunarity", (520, 10), 2.0, 4, 0.1, True)
swamps_slider = ravenui.Slider(ui, "Max Swamps", (622, 10), 10, 100, 0)
start_button = ravenui.Button(ui, "New Map", (10, 62), new_map, bg=(50, 200, 20))
update_button = ravenui.Button(ui, "Update Map", (112, 62), update_map, bg=(50, 200, 20))
update_button.hide()
noise_button = ravenui.Button(ui, "Perlin Noise", (214, 62), switch_noise, bg=(50, 200, 20))
tmx_button = ravenui.Button(ui, "Save TMX", (316, 62), make_tmx, bg=(50, 200, 20))
tmx_button.hide()

def draw():
    screen.blit(scale_map_surface, (10, 130))
    ui.draw()

# Game loop
running = True
while running:
    # keep loop running at the right speed
    clock.tick(FPS)
    # Process input (events)
    for event in pg.event.get():
        # check for closing window
        if event.type == pg.QUIT:
            running = False
        else: # Send events to the UI
            ui.events(event.type)
    # Update
    ui.update()

    # Draw / render
    screen.fill(BLACK)
    draw()
    pg.display.flip()
pg.quit()