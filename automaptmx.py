import ravenui
import numpy as np
import pygame as pg
import noise, cv2 # pip3 install opencv-python
import random
import math
import os
import shutil
from os import path
vec = pg.math.Vector2

program_folder = path.dirname(__file__)
masks_folder = path.join(program_folder, 'masks')
tmx_folder = path.join(program_folder, 'tmx_maps')

WIDTH = 732
HEIGHT = 800
MAP_IMG_SIZE = 608
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
STUMP_TILES = [324, 351, 378, 405]
RIVER_CHANCE = 15 # Higher number lower chance.
RIVER_DIRCHANCE = 10
RIVER_ROCKS = [636, 637, 638]
RIVER_ROCKS = [636, 637, 638, 639, 640]
WAVE_TILES = [231, 232, 233,258, 260, 285, 286, 287]
WAVE_CORNER_TILES = [237, 238, 480, 481]
WAVE_CORNER_TILES2 = [265, 264, 454, 453]
MAIN_LARGE_TREE_TILES = [649, 656, 838, 845]
ALL_LARGE_TREE_TILES = []
LARGE_TREE_DIMENSIONS = 7
LARGE_TREE_CENTER = 3
for tree in MAIN_LARGE_TREE_TILES:
    tile = tree
    for y in range(0, LARGE_TREE_DIMENSIONS):
        for x in range(0, LARGE_TREE_DIMENSIONS):
            ALL_LARGE_TREE_TILES.append(tile)
            tile += 1
        tile += (TILESET_COLUMNS - LARGE_TREE_DIMENSIONS)

MAIN_MED_TREE_TILES = [666, 671, 801, 806]
ALL_MED_TREE_TILES = []
MED_TREE_DIMENSIONS = 5
MED_TREE_CENTER = 2
for tree in MAIN_MED_TREE_TILES:
    tile = tree
    for y in range(0, MED_TREE_DIMENSIONS):
        for x in range(0, MED_TREE_DIMENSIONS):
            ALL_MED_TREE_TILES.append(tile)
            tile += 1
        tile += (TILESET_COLUMNS - MED_TREE_DIMENSIONS)

MAIN_SMALL_TREE_TILES = [663, 825, 744, 906]
ALL_SMALL_TREE_TILES = []
SMALL_TREE_DIMENSIONS = 3
SMALL_TREE_CENTER = 1
for tree in MAIN_SMALL_TREE_TILES:
    tile = tree
    for y in range(0, SMALL_TREE_DIMENSIONS):
        for x in range(0, SMALL_TREE_DIMENSIONS):
            ALL_SMALL_TREE_TILES.append(tile)
            tile += 1
        tile += (TILESET_COLUMNS - SMALL_TREE_DIMENSIONS)
EVERY_TREE_TILE = ALL_SMALL_TREE_TILES + ALL_MED_TREE_TILES + ALL_LARGE_TREE_TILES

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
DENS_FACTOR = 0
OCEAN_FACTOR = 0.35
SEALVL_FACTOR = 1

BLACK = [0, 0, 0]
colors = [[93, 120, 150], [108, 135, 168], [147, 174, 171], [165, 195, 228], [189, 219, 219], [261, 231, 174], [152, 118, 84], [102, 159, 69], [75, 141, 33], [102, 66, 45], [120, 105, 72], [123, 108, 84], [144, 144, 144], [168, 168, 168], [183, 183, 183], [225, 225, 243], [255, 255, 255], [240, 250, 255]]
swampcolors = [[0, 105, 150], [0, 125, 130], [0, 145, 110], [0, 160, 100], [0, 165, 75], [0, 170, 55], [0, 180, 55], [0, 190, 55], [0, 205, 50], [0, 205, 50], [0, 205, 50], [0, 205, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
RIVER_COLOR = [0, 0, 255]
tundra_colors = [[93, 120, 150], [108, 135, 168], [147, 174, 171], [165, 195, 228], [189, 219, 219], [261, 231, 254], [252, 118, 204], [182, 225, 229], [202, 255, 249], [102, 66, 45], [120, 105, 72], [123, 108, 84], [144, 144, 144], [168, 168, 168], [183, 183, 183], [225, 225, 243], [255, 255, 255], [240, 250, 255]]
tundra_swampcolors = [[0, 105, 255], [0, 125, 255], [0, 145, 255], [0, 160, 255], [0, 165, 255], [0, 170, 255], [0, 180, 255], [0, 190, 255], [0, 205, 255], [0, 205, 255], [0, 205, 255], [0, 205, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

def blur_mask(temp_mask):
    temp_img = temp_mask.astype('uint8')
    kernel = np.ones((3, 3), np.uint8)
    new_img = temp_img
    blur_img = cv2.GaussianBlur(temp_img, (335, 335), 33, cv2.BORDER_CONSTANT)
    for i, color in enumerate(range(254, 0, -6)):
        fact = math.ceil(i * i/10)
        dilate_img = temp_img.astype('uint8')
        dilate_img = cv2.dilate(dilate_img, kernel, iterations=fact)
        dilate_img = np.where(dilate_img > 0, color, 0)
        new_img = np.where(new_img == 0, dilate_img, new_img)
    new_img = np.where(blur_img == 0, 0, new_img) #gets rid of squarness before final blur.
    kernel = cv2.getGaussianKernel(55, 15, cv2.CV_32F)
    temp_img = new_img.astype('float32')
    temp_img = cv2.sepFilter2D(temp_img, cv2.CV_32F, kernel, kernel)
    return temp_img

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
scale_map_surface = pg.Surface((MAP_IMG_SIZE, MAP_IMG_SIZE)).convert()
overmap_size = int(math.sqrt(len(os.listdir(masks_folder))) * MAP_SIZE) # Takes the files in the masks folder and assumes they fit together in a square world.
overworld_noise = np.zeros((overmap_size, overmap_size))
world_noise = np.zeros((MAP_SIZE, MAP_SIZE))
random_plant_noise = np.zeros((MAP_SIZE, MAP_SIZE))
world = np.zeros((MAP_SIZE, MAP_SIZE))
river_edges = np.zeros((MAP_SIZE, MAP_SIZE), np.int64)
river_edges2 = np.zeros((MAP_SIZE, MAP_SIZE), np.int64)
deep_river_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)
swamps_arr = np.zeros((MAP_SIZE, MAP_SIZE), int) # Used for storing regions that can be turned into swamps.
ocean_regions = np.zeros((MAP_SIZE, MAP_SIZE), int) # Used to tell oceans/seas from lakes to add waves to.
beach_regions = np.zeros((MAP_SIZE, MAP_SIZE), int) # Used to tell were the beaches are.
desert_regions = np.zeros((MAP_SIZE, MAP_SIZE), int)
rivers = np.zeros((MAP_SIZE, MAP_SIZE), int) # Used for river layer.
img = cv2.imread("mapmask1.png", 0)
img = blur_mask(img)
mask = img / 255.0

noise_type = noise.pnoise2
noise_type2 = noise.pnoise2
scale = 100.0
octaves = 6
persistence = 0.5
lacunarity = 2.0
biome_type = 'Tropics'



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

def return_tile_id(tile):
    tileID = tile & ~(0x80000000 | 0x40000000 | 0x20000000)  # clear the flags
    return tileID

def make_river(tbase, y, x, dr, count = 0):
    global rivers, world_noise, swamps_arr, river_edges, river_edges2
    if (count < 900) and (tbase[y, x] > 5) and not ((swamps_arr[y, x] == 1) and (tbase[y, x] <= 8)):
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
                    else:
                        return
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

def toggle_biome():
    global biome_type
    if biome_type == "Tundra":
        biome_type = 'Tropics'
    else:
        biome_type = "Tundra"
    biome_button.update_text(biome_type)

def switch_noise():
    global noise_type
    if noise_type == noise.snoise2:
        noise_type = noise.pnoise2
        noise_txt = "Perlin Noise"
    elif noise_type == noise.pnoise2:
        noise_type = noise.snoise2
        noise_txt = "Simplex Noise"
    noise_button.update_text(noise_txt)

def switch_noise2():
    global noise_type2
    if noise_type2 == noise.snoise2:
        noise_type2 = noise.pnoise2
        noise_txt = "Perlin Noise"
    elif noise_type2 == noise.pnoise2:
        noise_type2 = noise.snoise2
        noise_txt = "Simplex Noise"
    mnoise_button.update_text(noise_txt)

def add_color(world):
    global colors, KINDSOFTILES, swampcolors, swamps_arr, tundra_colors, tundra_swampcolors, biome_type
    color_world = np.zeros(world.shape+(3,))
    if biome_type == "Tundra":
        color_list = tundra_colors
        scolor_list = tundra_swampcolors
    else:
        color_list = colors
        scolor_list = swampcolors

    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            for t in range(KINDSOFTILES):
                if t == KINDSOFTILES-1:
                    if world[i][j] < threshold + 1:
                        color_world[i][j] = color_list[t]
                elif world[i][j] < threshold + (t+1)/KINDSOFTILES:
                    if swamps_arr[i][j] == 1:
                        color_world[i][j] = scolor_list[t]
                    else:
                        color_world[i][j] = color_list[t]
                    break
                elif world[i][j] >= threshold + 1:
                    color_world[i][j] = color_list[KINDSOFTILES-1]
            if rivers[i][j]:
                color_world[i][j] = RIVER_COLOR
    return color_world

def make_random_mask():
    global mask, map_surface, scale_map_surface
    print("Making random mask...")
    random_mask = np.zeros((MAP_SIZE, MAP_SIZE), int)
    random_mask_noise = np.zeros((MAP_SIZE, MAP_SIZE))
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            random_mask_noise[i][j] = noise_type2(i/mscale_slider.val,
                                        j/mscale_slider.val,
                                        octaves=moctaves_slider.val,
                                        persistence=mpersistence_slider.val,
                                        lacunarity=mlacunarity_slider.val,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=0) + 0.5  # The 0.5 makes i return values between 0 and 1.
    # Creates a random island cluster mask
    random_img = np.where(random_mask_noise > scope_slider.val, 255, 0)
    random_img = random_img.astype('uint8')
    shade = 0
    padding = padding_slider.val
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if random_img[x, y] == 255:
                shade += 1
                if shade > 255:
                    shade = 255
                    break
                cv2.floodFill(random_img, None, (y, x), shade)

    islands_list = [] #Makes a list of randomly selectable islands
    for color in range(0, shade + 1):
        rmask_arr = np.where(random_img == color, 255, 0)
        if min_island_size_slider.val < np.count_nonzero(rmask_arr == 255) < max_island_size_slider.val:
            add_island = True
            for pad in range(0, padding): # Makes sure the island is not on the map edge.
                if True in [rmask_arr.any(axis=1)[pad], rmask_arr.any(axis=1)[MAP_SIZE - pad - 1], rmask_arr.any(axis=0)[pad], rmask_arr.any(axis=0)[MAP_SIZE - pad - 1]]:
                    add_island = False
            if add_island:
                islands_list.append(rmask_arr)

    max_islands = max_islands_slider.val
    if len(islands_list) < max_islands_slider.val:
        max_islands = len(islands_list)
    if max_islands == 0:
        print("Error: No islands with selected mask settings. Alter settings and try again.")
        return
    for i in range(0, max_islands):
        selected_isl = random.choice(range(0, len(islands_list)))
        random_mask = np.add(random_mask, islands_list[selected_isl])
        del islands_list[selected_isl]

    #Displays mask on screen.
    random_img = random_mask.astype('uint8')
    size = random_img.shape[1::-1]
    random_img= np.repeat(random_img.reshape(size[1], size[0], 1), 3, axis = 2)
    new_mask = np.uint8(random_img)
    pg.surfarray.blit_array(map_surface, new_mask)
    temp_surf = pg.transform.rotate(map_surface, -90)
    temp_surf2 = pg.transform.flip(temp_surf, True, False)
    scale_map_surface = pg.transform.scale(temp_surf2, (MAP_IMG_SIZE, MAP_IMG_SIZE))
    cv2.imwrite("random_mask.png", random_img)
    print("Map mask complete.")
    draw()

    random_img = blur_mask(random_mask)
    mask = random_img / 255.0


def auto_world_map():  # Used for creating a world with numberous maps in the masks directory.
    global mask, biome_type, map_surface, scale_map_surface, masks_folder, tmx_folder, random_plant_noise, rivers, river_edges, river_edges2, deep_river_arr, swamps_arr, ocean_regions, beach_regions
    generate_overworld_noise()
    tundra_maps_list = [12]  # Used for world maps with north pole in center and south pole surrounding the edges.
    just_water_list = [0, 1, 3, 4, 7, 8, 10, 11, 14, 16, 17, 18, 19, 20, 23]
    already_finished = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 23]
    #for i in range(0, 12):
    #    already_finished.append(i)

    fromlist_button.update_text('0%')
    ui.draw()
    pg.display.flip()
    number_of_files = len([name for name in os.listdir(masks_folder) if os.path.isfile(os.path.join(masks_folder, name))])
    for i in range(0, number_of_files):
        percent_complete = int(i/number_of_files * 100)
        fromlist_button.update_text(str(percent_complete) +'%')
        if i not in already_finished:
            random_plant_noise = np.zeros((MAP_SIZE, MAP_SIZE))
            river_edges = np.zeros((MAP_SIZE, MAP_SIZE), np.int64)
            river_edges2 = np.zeros((MAP_SIZE, MAP_SIZE), np.int64)
            deep_river_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)
            swamps_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used for storing regions that can be turned into swamps.
            ocean_regions = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used to tell oceans/seas from lakes to add waves to.
            beach_regions = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used to tell were the beaches are.
            desert_regions = np.zeros((MAP_SIZE, MAP_SIZE), int)
            rivers = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used for river layer.
            file = '{}.png'.format(i)
            name = file.replace('.png', '')
            if int(name) in tundra_maps_list:
                biome_type = 'Tundra'
            else:
                biome_type = 'Tropics'
            file_path = path.join(masks_folder, file)
            img = cv2.imread(file_path, 0)
            img = blur_mask(img)
            mask = img / 255.0

            # Displays mask on screen.
            pg.surfarray.blit_array(map_surface, img)
            temp_surf = pg.transform.rotate(map_surface, -90)
            temp_surf2 = pg.transform.flip(temp_surf, True, False)
            scale_map_surface = pg.transform.scale(temp_surf2, (MAP_IMG_SIZE, MAP_IMG_SIZE))
            draw()
            if np.all((mask == 0)): # If the image is blank then it just copies an existing map that is all water.
                dest_dir = tmx_folder
                src_dir = os.path.join(tmx_folder, 'water')
                src_file = os.path.join(src_dir, 'just_water.tmx')
                shutil.copy(src_file, dest_dir)
                dst_file = os.path.join(dest_dir, 'just_water.tmx')
                new_dst_file_name = os.path.join(dest_dir, '{}.tmx'.format(i))
                os.rename(dst_file, new_dst_file_name)  # rename
            else:
                new_map(True, i)
                draw()
                file = file.replace("png", "tmx")
                output_path = path.join(tmx_folder, file)
                make_tmx(output_path)
    fromlist_button.update_text('Auto World')

def generate_overworld_noise():
    global overworld_noise, overmap_size
    print("Building new overworld map...")
    # Noise settings
    # Creates a perlin noise array the size of the map.
    overworld = np.zeros((overmap_size, overmap_size))
    roct = random.randrange(4, 12)
    rscale = random.randrange(40, 75)
    for i in range(overmap_size):
        for j in range(overmap_size):
            overworld[i][j] = noise_type(i/scale_slider.val,
                                        j/scale_slider.val,
                                        octaves=octaves_slider.val,
                                        persistence=persistence_slider.val,
                                        lacunarity=lacunarity_slider.val) + 0.5  # The 0.5 makes i return values between 0 and 1.
    overworld_noise = overworld

def new_map(overmap = False, mapnum = 0):
    global scale_map_surface, world, mask, swamps_mask, random_plant_noise, random_mask, overmap_size, overworld_noise
    start_button.hide()
    print("Building new map...")
    # Noise settings
    # Creates a perlin noise array the size of the map.
    roct = random.randrange(4, 12)
    rscale = random.randrange(40, 75)
    if not overmap:
        world = np.zeros((MAP_SIZE, MAP_SIZE))
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
    else: #Only slices out a map sized section of the overworld noise data. This makes sure the noise is consistent between maps.
        dimension = overmap_size/MAP_SIZE
        print(dimension)
        map_row = int(mapnum // dimension)
        map_col = int(mapnum - dimension*(mapnum//dimension))
        print(map_row, map_col)
        world = overworld_noise[map_row * MAP_SIZE: map_row  * MAP_SIZE + MAP_SIZE, map_col * MAP_SIZE: map_col * MAP_SIZE + MAP_SIZE]
        print(map_row * MAP_SIZE, map_row  * MAP_SIZE + MAP_SIZE, map_col * MAP_SIZE, map_col * MAP_SIZE + MAP_SIZE)

        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
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
    global world_noise, scale_map_surface, world, mask, swamps_arr, MAX_SWAMP_SIZE, ocean_regions, beach_regions, rivers, deep_river_arr, desert_regions
    swamps_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used for storing regions that can be turned into swamps.
    snow_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)
    ocean_regions = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used to tell oceans/seas from lakes to add waves to.
    beach_regions = np.zeros((MAP_SIZE, MAP_SIZE), int)  # Used to tell were the beaches are.
    desert_regions = np.zeros((MAP_SIZE, MAP_SIZE), int)
    rivers = np.zeros((MAP_SIZE, MAP_SIZE), int)
    deep_river_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)
    print('Updating map...')
    # Combines mask image data with perlin noise
    world_noise = np.zeros_like(world)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            world_noise[i][j] = ((world[i][j]*(1-mask_slider.val)) + (mask[i][j]*mask_slider.val) - ocean_slider.val)
    # get it between 0 and 1
    max_grad = np.amax(world_noise)
    if max_grad > 0:
        world_noise = world_noise / max_grad
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if elevation_slider.val != 1: #Changes elevation steepness.
                world_noise[i][j] *= elevation_slider.val
    world_noise = np.where(world_noise < 0, 0, world_noise)  # Gets rid of negatives.
    world_noise = np.where(world_noise > 0.98, 0.98, world_noise)  # Gets rid of values above 1ish.
    world_noise = np.where(np.isnan(world_noise), 0, world_noise) # Removes NaN values.

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
    for color in range(0, shade + 1):
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

    # Finds ocean areas and potential deserts/lakes.
    pot_deserts = []
    for color in range(0, shade + 1):
        ocean_arr = np.where(ocean_img == color, 1, 0)
        if np.count_nonzero(ocean_arr == 1) > MIN_OCEAN_SIZE:
            ocean_regions = np.add(ocean_regions, ocean_arr) # Finds the ocean tiles to add waves to
        else:
            pot_deserts.append(ocean_arr)
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

    # Creates deserts
    deserts_added = 0
    while (deserts_added < deserts_slider.val) and (deserts_added < len(pot_deserts)):
        for area in pot_deserts:
            if deserts_added < deserts_slider.val:
                if random.randrange(0, 2) == 1:
                    desert_regions = np.add(desert_regions, area)
                    deserts_added += 1
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if desert_regions[i, j]:
                world_noise[i, j] = 0.3


    # Creates rivers
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if snow_arr[i][j] == 1:
                count = [0, 0, 0, 0]
                # Marks first tiles for river heads then creates rivers where they flow to the ocean or lakes.
                try:
                    if temp_array[i - 1][j] == (temp_array[i][j] - 1):
                        if random.randrange(0, river_slider.val) == 1:
                            make_river(temp_array, i, j, 1)
                    if temp_array[i + 1][j] == (temp_array[i][j] - 1):
                        if random.randrange(0, river_slider.val) == 1:
                            make_river(temp_array, i, j, 0)
                    if temp_array[i][j - 1] == (temp_array[i][j] - 1):
                        if random.randrange(0, river_slider.val) == 1:
                            make_river(temp_array, i, j, 3)
                    if temp_array[i][j + 1] == (temp_array[i][j] - 1):
                        if random.randrange(0, river_slider.val) == 1:
                            make_river(temp_array, i, j, 2)
                except:
                    pass

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
    scale_map_surface = pg.transform.scale(temp_surf2, (MAP_IMG_SIZE, MAP_IMG_SIZE))
    print("Done")
    tmx_button.show()

# Creates TMX file based off of noise+mask
def make_tmx(filename = path.join(tmx_folder, "newmap.tmx")):
    global world_noise, random_plant_noise, rivers, river_edges, deep_river_arr, biome_type
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
    corners_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), np.int64)
    overlay_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), np.int64) #used for corners that overlay water edges.
    overlay2_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), np.int64) #used for corners that overlay water edges.
    wave_val_arr = np.zeros((MAP_SIZE + 2, MAP_SIZE + 2), np.int64)
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
    for y in range(2, MAP_SIZE):
        for x in range(2, MAP_SIZE):
            # Scans map 4 blocks at a time to find the corner pattern.
            find_list = [arr1[y][x], arr1[y+1][x], arr1[y][x+1], arr1[y+1][x+1]] # Makes a list of 4 tile block values. Checks for basic corners.
            if (arr1[y][x] == 0) and (arr1[y][x+1] > 0) and (river_edges[y - 1][x - 1] == 0):
                river_edges[y - 1][x - 1] = hflip(RIVER_ROCKS[1])
                if return_tile_id(overlay2_layer_vals[y - 1][x - 1]) in [168, 114, 87, 189, 135, 108]: # Gets rid of sand overlays on rivers.
                    overlay2_layer_vals[y - 1][x - 1] = 0
            if (arr1[y][x] > 0) and (arr1[y][x+1] == 0) and (river_edges[y - 1][x] == 0):
                river_edges[y - 1][x] = RIVER_ROCKS[1]
                if return_tile_id(overlay2_layer_vals[y - 1][x]) in [168, 114, 87, 189, 135, 108]: # Gets rid of sand overlays on rivers.
                    overlay2_layer_vals[y - 1][x] = 0
            if (arr1[y][x] == 0) and (arr1[y + 1][x] > 0) and (river_edges[y - 1][x - 1] == 0):
                river_edges[y - 1][x - 1] = vdflip(RIVER_ROCKS[1])
                if return_tile_id(overlay2_layer_vals[y - 1][x - 1]) in [168, 114, 87, 189, 135, 108]: # Gets rid of sand overlays on rivers.
                    overlay2_layer_vals[y - 1][x - 1] = 0
            if (arr1[y][x] > 0) and (arr1[y + 1][x] == 0) and (river_edges[y][x - 1] == 0):
                river_edges[y][x - 1] = dflip(RIVER_ROCKS[1])
                if return_tile_id(overlay2_layer_vals[y][x - 1]) in [168, 114, 87, 189, 135, 108]: # Gets rid of sand overlays on rivers.
                    overlay2_layer_vals[y][x - 1] = 0

            if 0 not in find_list:
                if not all(element == find_list[0] for element in find_list): # Checks to see if all river elements are the same if not changes them into lake water.
                    rivers[y - 1][x - 1] = rivers[y][x - 1] = rivers[y - 1][x] = rivers[y][x] = LAKE_SHALLOWS
            if find_list.count(0) == 1:
                if rivers[y - 1][x - 1] > 0:
                    rivers[y - 1][x - 1] = LAKE_SHALLOWS
                elif 0 in [river_edges[y - 1][x - 1], river_edges2[y - 1][x - 1]]:
                        river_edges[y - 1][x - 1] = hflip(RIVER_ROCKS[1])
                        river_edges2[y - 1][x - 1] = vdflip(RIVER_ROCKS[1])
                if rivers[y][x - 1] > 0:
                    rivers[y][x - 1] = LAKE_SHALLOWS
                elif 0 in [river_edges[y][x - 1], river_edges2[y][x - 1]]:
                    river_edges[y][x - 1] = hflip(RIVER_ROCKS[1])
                    river_edges2[y][x - 1] = dflip(RIVER_ROCKS[1])
                if rivers[y - 1][x] > 0:
                    rivers[y - 1][x] = LAKE_SHALLOWS
                elif 0 in [river_edges[y - 1][x], river_edges2[y - 1][x]]:
                    river_edges[y - 1][x] = RIVER_ROCKS[1]
                    river_edges2[y - 1][x] = vdflip(RIVER_ROCKS[1])
                if rivers[y][x] > 0:
                    rivers[y][x] = LAKE_SHALLOWS
                elif 0 in [river_edges[y][x], river_edges2[y][x]]:
                    river_edges[y][x] = RIVER_ROCKS[1]
                    river_edges2[y][x] = dflip(RIVER_ROCKS[1])
            if find_list.count(0) == 3: # Adds river rock edges.
                if rivers[y - 1][x - 1] > 0:
                    if river_edges[y][x] == 0:
                        river_edges[y][x] = RIVER_ROCKS[2]
                    else:
                        river_edges2[y][x] = RIVER_ROCKS[2]
                if rivers[y][x - 1] > 0:
                    if river_edges[y - 1][x] == 0:
                        river_edges[y - 1][x] = vflip(RIVER_ROCKS[2])
                    else:
                        river_edges2[y - 1][x] = vflip(RIVER_ROCKS[2])
                if rivers[y - 1][x] > 0:
                    if river_edges[y][x - 1] == 0:
                        river_edges[y][x - 1] = hflip(RIVER_ROCKS[2])
                    else:
                        river_edges2[y][x - 1] = hflip(RIVER_ROCKS[2])
                if rivers[y][x] > 0:
                    if river_edges[y - 1][x - 1] == 0:
                        river_edges[y - 1][x - 1] = hvflip(RIVER_ROCKS[2])
                    else:
                        river_edges2[y - 1][x - 1] = hvflip(RIVER_ROCKS[2])


    plant_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), int)
    tree_layer_vals = np.zeros((MAP_SIZE, MAP_SIZE), int)
    stump = None
    tree_dims = [SMALL_TREE_DIMENSIONS, MED_TREE_DIMENSIONS, LARGE_TREE_DIMENSIONS]
    tree_centers = [SMALL_TREE_CENTER, MED_TREE_CENTER, LARGE_TREE_CENTER]
    tree_tiles = [ALL_SMALL_TREE_TILES, ALL_MED_TREE_TILES, ALL_LARGE_TREE_TILES]
    main_tree_tiles = [MAIN_SMALL_TREE_TILES, MAIN_MED_TREE_TILES, MAIN_LARGE_TREE_TILES]
    # Places vegetation
    print("Addinig plants...")
    for y in range(0, MAP_SIZE):
        for x in range(0, MAP_SIZE):
            if rivers[y][x] == 0:
                tree_size = random.randrange(0, 3)
                if (ocean_regions[y][x] == 0) and (y < MAP_SIZE - tree_dims[tree_size]) and (x < MAP_SIZE - tree_dims[tree_size]): # Keeps it for going out of range.
                    if random.randrange(0, 10) and random_plant_noise[y][x]: # Only does tree checks every 10th tile on average.
                        if tree_size == 2:
                            tree_scope = [tree_layer_vals[y][x], tree_layer_vals[y][x+1], tree_layer_vals[y][x+2], tree_layer_vals[y][x+3], tree_layer_vals[y][x+4], tree_layer_vals[y][x+5], tree_layer_vals[y][x+6],
                                          tree_layer_vals[y+1][x], tree_layer_vals[y+1][x+1], tree_layer_vals[y+1][x+2], tree_layer_vals[y+1][x+3], tree_layer_vals[y+1][x+4], tree_layer_vals[y+1][x+5], tree_layer_vals[y+1][x+6],
                                          tree_layer_vals[y +2][x], tree_layer_vals[y +2][x+1], tree_layer_vals[y +2][x+2], tree_layer_vals[y +2][x+3], tree_layer_vals[y +2][x+4], tree_layer_vals[y +2][x+5], tree_layer_vals[y +2][x+6],
                                          tree_layer_vals[y +3][x], tree_layer_vals[y +3][x+1], tree_layer_vals[y +3][x+2], tree_layer_vals[y +3][x+3], tree_layer_vals[y +3][x+4], tree_layer_vals[y +3][x+5], tree_layer_vals[y +3][x+6],
                                          tree_layer_vals[y +4][x], tree_layer_vals[y +4][x+1], tree_layer_vals[y +4][x+2], tree_layer_vals[y +4][x+3], tree_layer_vals[y +4][x+4], tree_layer_vals[y +4][x+5], tree_layer_vals[y +4][x+6],
                                          tree_layer_vals[y +5][x], tree_layer_vals[y +5][x+1], tree_layer_vals[y +5][x+2], tree_layer_vals[y +5][x+3], tree_layer_vals[y +5][x+4], tree_layer_vals[y +5][x+5], tree_layer_vals[y +5][x+6],
                                          tree_layer_vals[y +6][x], tree_layer_vals[y +6][x+1], tree_layer_vals[y +6][x+2], tree_layer_vals[y +6][x+3], tree_layer_vals[y +6][x+4], tree_layer_vals[y +6][x+5], tree_layer_vals[y +6][x+6]]
                        elif tree_size == 1:
                            tree_scope = [tree_layer_vals[y][x], tree_layer_vals[y][x+1], tree_layer_vals[y][x+2], tree_layer_vals[y][x+3], tree_layer_vals[y][x+4],
                                          tree_layer_vals[y+1][x], tree_layer_vals[y+1][x+1], tree_layer_vals[y+1][x+2], tree_layer_vals[y+1][x+3], tree_layer_vals[y+1][x+4],
                                          tree_layer_vals[y +2][x], tree_layer_vals[y +2][x+1], tree_layer_vals[y +2][x+2], tree_layer_vals[y +2][x+3], tree_layer_vals[y +2][x+4],
                                          tree_layer_vals[y +3][x], tree_layer_vals[y +3][x+1], tree_layer_vals[y +3][x+2], tree_layer_vals[y +3][x+3], tree_layer_vals[y +3][x+4],
                                          tree_layer_vals[y +4][x], tree_layer_vals[y +4][x+1], tree_layer_vals[y +4][x+2], tree_layer_vals[y +4][x+3], tree_layer_vals[y +4][x+4]]
                        else:
                            tree_scope = [tree_layer_vals[y][x], tree_layer_vals[y][x+1], tree_layer_vals[y][x+2],
                                          tree_layer_vals[y+1][x], tree_layer_vals[y+1][x+1], tree_layer_vals[y+1][x+2],
                                          tree_layer_vals[y +2][x], tree_layer_vals[y +2][x+1], tree_layer_vals[y +2][x+2]]

                        # Trees
                        place_tree = True
                        if beach_regions[y][x] and (6 < base_layer_vals[y][x] < 11):
                            tree = main_tree_tiles[tree_size][2]
                            stump = STUMP_TILES[2]
                            prob = PALM_PROB
                        elif (not beach_regions[y][x]) and base_layer_vals[y][x] in [8, 9]:
                            tree = main_tree_tiles[tree_size][1]
                            stump = STUMP_TILES[1]
                            prob = GREENTREE_PROB
                        elif (not beach_regions[y][x]) and base_layer_vals[y][x] in [10, 11]:
                            if random.randrange(0, 3) == 1:
                                tree = main_tree_tiles[tree_size][3]
                                stump = STUMP_TILES[3]
                                prob = PINE_PROB
                            elif random.randrange(0, 10) == 1:
                                tree = main_tree_tiles[tree_size][0]
                                stump = STUMP_TILES[0]
                                prob = DEADTREE_PROB
                            else:
                                tree = main_tree_tiles[tree_size][1]
                                stump = STUMP_TILES[1]
                                prob = GREENTREE_PROB
                        elif (not beach_regions[y][x]) and (11 < base_layer_vals[y][x] < 14):
                            if random.randrange(0, 10) == 1:
                                tree = main_tree_tiles[tree_size][0]
                                stump = STUMP_TILES[0]
                                prob = DEADTREE_PROB
                            else:
                                tree = main_tree_tiles[tree_size][3]
                                stump = STUMP_TILES[3]
                                prob = PINE_PROB * 2
                        elif (not beach_regions[y][x]) and (14 < base_layer_vals[y][x] < 17):
                            if random.randrange(0, 10) == 1:
                                tree = main_tree_tiles[tree_size][0]
                                stump = STUMP_TILES[0]
                                prob = DEADTREE_PROB * 4
                            else:
                                tree = main_tree_tiles[tree_size][3]
                                stump = STUMP_TILES[3]
                                prob = PINE_PROB * 4
                        else:
                            place_tree = False
                        if place_tree:
                            prob = int(prob / plant_slider.val)
                        if place_tree and (random.randrange(0, prob * 10) == 1):
                            for tile in tree_scope:
                                if tile in EVERY_TREE_TILE: # checks to see if there are already tree tiles before placing a tree.
                                    place_tree = False
                                    continue
                            if place_tree:
                                tile = tree
                                for i in range(0, tree_dims[tree_size]):
                                    for j in range(0, tree_dims[tree_size]):
                                        tree_layer_vals[y + i][x + j] = tile
                                        tile += 1
                                        if (i == tree_centers[tree_size]) and (j == tree_centers[tree_size]):
                                            plant_layer_vals[y + i][x + j] = stump
                                    tile += (TILESET_COLUMNS - tree_dims[tree_size])


                # Plants
                if plant_layer_vals[y][x] == 0:
                    if base_layer_vals[y][x] == 8: # If grassland.
                        if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                            for plant, prob in GRASSLAND_PLANTS.items():
                                prob = int(prob / plant_slider.val)
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
                                prob = int(prob / plant_slider.val)
                                if pfact*prob > 1:
                                    if random.randrange(0, pfact*prob) == 1:
                                        corners_layer_vals[y][x] = plant # Puts them under the water.
                    elif base_layer_vals[y][x] == 10:  # Forest
                        if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                            for plant, prob in FOREST_PLANTS.items():
                                prob = int(prob / plant_slider.val)
                                if random.randrange(0, FPLANT_FACT*prob) == 1:
                                    plant_layer_vals[y][x] = plant
                    elif water_layer_vals[y][x] in [SWAMP_SHALLOWS]:  # Swamp water
                        if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                            for plant, prob in SWAMPW_PLANTS.items():
                                prob = int(prob / plant_slider.val)
                                if random.randrange(0, SPLANTW_FACT*prob) == 1:
                                    plant_layer_vals[y][x] = plant
                    elif base_layer_vals[y][x] in [26, 27]:  # swamp (not in water)
                        if corners_layer_vals[y][x] == 0:
                            for plant, prob in SWAMPD_PLANTS.items():
                                prob = int(prob / plant_slider.val)
                                if random.randrange(0, SPLANTD_FACT * prob) == 1:
                                    plant_layer_vals[y][x] = plant
                    elif (base_layer_vals[y][x] == 6) and (not beach_regions[y][x]) and desert_regions[y, x]:  # desert
                        if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                            for plant, prob in DESERT_PLANTS.items():
                                prob = int(prob / plant_slider.val)
                                if random.randrange(0, DPLANT_FACT*prob) == 1:
                                    plant_layer_vals[y][x] = plant
                    elif (base_layer_vals[y][x] == 7) and (not beach_regions[y][x]):  # dirt not near beaches
                        if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                            for plant, prob in WASTELAND_PLANTS.items():
                                prob = int(prob / plant_slider.val)
                                if random.randrange(0, WPLANT_FACT*prob) == 1:
                                    plant_layer_vals[y][x] = plant
                    elif 10 < base_layer_vals[y][x] < 17: # If mountains
                        if (corners_layer_vals[y][x] == 0) and random_plant_noise[y][x]:
                            pfact = (base_layer_vals[y][x] - 10) * MPLANT_FACT
                            for plant, prob in MOUNTAIN_PLANTS.items():
                                prob = int(prob / plant_slider.val)
                                if random.randrange(0, pfact*prob) == 1:
                                    plant_layer_vals[y][x] = plant

    # Makes deep water areas in rivers
    rivers = np.where(deep_river_arr == 0, rivers, LAKE_WATER) # Creates deep water in rivers, ponds and lakes.

    # Gets rid of flowing river edges along lakes, ponds and wide rivers
    arr1 = np.pad(rivers, pad_width=1, mode='constant', constant_values=0)
    # I know I screwed up and my x's and y's are switched in this loop.... But it works.
    for y in range(1, MAP_SIZE + 1):
        for x in range(1, MAP_SIZE + 1):
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

            try: # Puts edges in patches with water nothing then water
                if arr1[y][x] and arr1[y][x + 2] and not arr1[y][x + 1]:
                    river_edges[y - 1][x] = RIVER_ROCKS[1]
                    river_edges2[y - 1][x] = hflip(RIVER_ROCKS[1])
                if arr1[y][x] and arr1[y + 2][x] and not arr1[y + 1][x]:
                    river_edges[y][x - 1] = dflip(RIVER_ROCKS[1])
                    river_edges2[y][x - 1] = vdflip(RIVER_ROCKS[1])
                # Fixes corners next to edges.
                if arr1[y][x] and (return_tile_id(river_edges[y - 1][x]) == RIVER_ROCKS[2]):
                    river_edges2[y - 1][x] = RIVER_ROCKS[1]
                if arr1[y][x] and (return_tile_id(river_edges[y - 1][x - 2]) == RIVER_ROCKS[2]):
                    river_edges2[y - 1][x - 2] = hflip(RIVER_ROCKS[1])
                if arr1[y][x] and (return_tile_id(river_edges[y][x - 1]) == RIVER_ROCKS[2]):
                    river_edges2[y][x - 1] = dflip(RIVER_ROCKS[1])
                if arr1[y][x] and (return_tile_id(river_edges[y - 2][x - 1]) == RIVER_ROCKS[2]):
                    river_edges2[y - 2][x - 1] = vdflip(RIVER_ROCKS[1])
            except:
                pass

    overlay2_layer_vals = np.where(rivers == 0, overlay2_layer_vals, 0) # Gets rid of overlays on top of rivers.
    river_edges = np.where(rivers == 0, river_edges, RIVER_ROCKS[0])
    overlay_layer_vals = np.where(river_edges == 0, overlay_layer_vals, river_edges)
    overlay2_layer_vals = np.where(river_edges2 == 0, overlay2_layer_vals, river_edges2)

    # Merges layers down to reduce layer complexity on final tmx map: This helps reduce loading times considerably.
    plant_layer_vals = np.where(plant_layer_vals == 0, waves_layer_vals, plant_layer_vals)
    plant_layer_vals = np.where(plant_layer_vals == 0, rivers, plant_layer_vals)
    plant_layer_vals = np.where(plant_layer_vals == 0, overlay2_layer_vals, plant_layer_vals)
    overlay_layer_vals = np.where(overlay_layer_vals == 0, corners_layer_vals, overlay_layer_vals)
    corners_layer_vals = overlay_layer_vals

    if biome_type == 'Tundra':
        tile_set = 'tundratiles.png'
    else:
        tile_set = 'tropicstiles.png'
    # Writes base layer
    print("Writing base layer...")
    outfile = open(filename, "w")
    header = """<?xml version="1.0" encoding="UTF-8"?>     
    <map version="1.5" tiledversion="1.7.2" orientation="orthogonal" renderorder="right-down" width="{mapw}" height="{mapw}" tilewidth="32" tileheight="32" infinite="0" nextlayerid="6" nextobjectid="1">
     <tileset firstgid="1" name="automaptiles" tilewidth="32" tileheight="32" tilecount="{tile_count}" columns="{tileset_columns}">
      <image source="{tilesheet}" width="864" height="1216"/>
      <tile id="216">
       <animation>
        <frame tileid="216" duration="100"/>
        <frame tileid="243" duration="100"/>
        <frame tileid="297" duration="100"/>
        <frame tileid="324" duration="100"/>
        <frame tileid="351" duration="100"/>
        <frame tileid="378" duration="100"/>
        <frame tileid="405" duration="100"/>
        <frame tileid="432" duration="100"/>
        <frame tileid="459" duration="100"/>
        <frame tileid="486" duration="100"/>
        <frame tileid="513" duration="100"/>
        <frame tileid="540" duration="100"/>
        <frame tileid="567" duration="100"/>
        <frame tileid="594" duration="100"/>
        <frame tileid="621" duration="100"/>
       </animation>
      </tile>
      <tile id="217">
       <animation>
        <frame tileid="217" duration="100"/>
        <frame tileid="244" duration="100"/>
        <frame tileid="271" duration="100"/>
        <frame tileid="298" duration="100"/>
        <frame tileid="325" duration="100"/>
        <frame tileid="352" duration="100"/>
        <frame tileid="379" duration="100"/>
        <frame tileid="406" duration="100"/>
        <frame tileid="433" duration="100"/>
        <frame tileid="460" duration="100"/>
        <frame tileid="487" duration="100"/>
        <frame tileid="514" duration="100"/>
        <frame tileid="541" duration="100"/>
        <frame tileid="568" duration="100"/>
        <frame tileid="595" duration="100"/>
        <frame tileid="622" duration="100"/>
       </animation>
      </tile>
      <tile id="218">
       <animation>
        <frame tileid="218" duration="100"/>
        <frame tileid="245" duration="100"/>
        <frame tileid="272" duration="100"/>
        <frame tileid="299" duration="100"/>
        <frame tileid="326" duration="100"/>
        <frame tileid="353" duration="100"/>
        <frame tileid="380" duration="100"/>
        <frame tileid="407" duration="100"/>
        <frame tileid="434" duration="100"/>
        <frame tileid="461" duration="100"/>
        <frame tileid="488" duration="100"/>
        <frame tileid="515" duration="100"/>
        <frame tileid="542" duration="100"/>
        <frame tileid="569" duration="100"/>
        <frame tileid="596" duration="100"/>
        <frame tileid="623" duration="100"/>
       </animation>
      </tile>
      <tile id="219">
       <animation>
        <frame tileid="219" duration="100"/>
        <frame tileid="246" duration="100"/>
        <frame tileid="273" duration="100"/>
        <frame tileid="300" duration="100"/>
        <frame tileid="327" duration="100"/>
        <frame tileid="354" duration="100"/>
        <frame tileid="381" duration="100"/>
        <frame tileid="408" duration="100"/>
        <frame tileid="435" duration="100"/>
        <frame tileid="462" duration="100"/>
        <frame tileid="489" duration="100"/>
        <frame tileid="516" duration="100"/>
        <frame tileid="543" duration="100"/>
        <frame tileid="570" duration="100"/>
        <frame tileid="597" duration="100"/>
        <frame tileid="624" duration="100"/>
       </animation>
      </tile>
      <tile id="220">
       <animation>
        <frame tileid="220" duration="100"/>
        <frame tileid="247" duration="100"/>
        <frame tileid="274" duration="100"/>
        <frame tileid="301" duration="100"/>
        <frame tileid="328" duration="100"/>
        <frame tileid="355" duration="100"/>
        <frame tileid="382" duration="100"/>
        <frame tileid="409" duration="100"/>
        <frame tileid="436" duration="100"/>
        <frame tileid="463" duration="100"/>
        <frame tileid="490" duration="100"/>
        <frame tileid="517" duration="100"/>
        <frame tileid="544" duration="100"/>
        <frame tileid="571" duration="100"/>
        <frame tileid="598" duration="100"/>
        <frame tileid="625" duration="100"/>
       </animation>
      </tile>
      <tile id="221">
       <animation>
        <frame tileid="221" duration="100"/>
        <frame tileid="248" duration="100"/>
        <frame tileid="275" duration="100"/>
        <frame tileid="302" duration="100"/>
        <frame tileid="329" duration="100"/>
        <frame tileid="356" duration="100"/>
        <frame tileid="383" duration="100"/>
        <frame tileid="410" duration="100"/>
        <frame tileid="437" duration="100"/>
        <frame tileid="464" duration="100"/>
        <frame tileid="491" duration="100"/>
        <frame tileid="518" duration="100"/>
        <frame tileid="545" duration="100"/>
        <frame tileid="572" duration="100"/>
        <frame tileid="599" duration="100"/>
        <frame tileid="626" duration="100"/>
       </animation>
      </tile>
      <tile id="222">
       <animation>
        <frame tileid="222" duration="100"/>
        <frame tileid="249" duration="100"/>
        <frame tileid="276" duration="100"/>
        <frame tileid="303" duration="100"/>
        <frame tileid="330" duration="100"/>
        <frame tileid="357" duration="100"/>
        <frame tileid="384" duration="100"/>
        <frame tileid="411" duration="100"/>
       </animation>
      </tile>
      <tile id="223">
       <animation>
        <frame tileid="223" duration="100"/>
        <frame tileid="250" duration="100"/>
        <frame tileid="277" duration="100"/>
        <frame tileid="304" duration="100"/>
        <frame tileid="331" duration="100"/>
        <frame tileid="358" duration="100"/>
        <frame tileid="385" duration="100"/>
        <frame tileid="412" duration="100"/>
       </animation>
      </tile>
      <tile id="224">
       <animation>
        <frame tileid="224" duration="100"/>
        <frame tileid="251" duration="100"/>
        <frame tileid="278" duration="100"/>
        <frame tileid="305" duration="100"/>
        <frame tileid="332" duration="100"/>
        <frame tileid="359" duration="100"/>
        <frame tileid="386" duration="100"/>
        <frame tileid="413" duration="100"/>
       </animation>
      </tile>
      <tile id="225">
       <animation>
        <frame tileid="225" duration="100"/>
        <frame tileid="252" duration="100"/>
        <frame tileid="279" duration="100"/>
        <frame tileid="306" duration="100"/>
        <frame tileid="333" duration="100"/>
        <frame tileid="360" duration="100"/>
        <frame tileid="387" duration="100"/>
        <frame tileid="414" duration="100"/>
       </animation>
      </tile>
      <tile id="226">
       <animation>
        <frame tileid="226" duration="100"/>
        <frame tileid="253" duration="100"/>
        <frame tileid="280" duration="100"/>
        <frame tileid="307" duration="100"/>
        <frame tileid="334" duration="100"/>
        <frame tileid="361" duration="100"/>
        <frame tileid="388" duration="100"/>
        <frame tileid="415" duration="100"/>
       </animation>
      </tile>
      <tile id="227">
       <animation>
        <frame tileid="227" duration="100"/>
        <frame tileid="254" duration="100"/>
        <frame tileid="281" duration="100"/>
        <frame tileid="308" duration="100"/>
        <frame tileid="335" duration="100"/>
        <frame tileid="362" duration="100"/>
        <frame tileid="389" duration="100"/>
        <frame tileid="416" duration="100"/>
       </animation>
      </tile>
      <tile id="228">
       <animation>
        <frame tileid="228" duration="100"/>
        <frame tileid="255" duration="100"/>
        <frame tileid="282" duration="100"/>
        <frame tileid="309" duration="100"/>
        <frame tileid="336" duration="100"/>
        <frame tileid="363" duration="100"/>
        <frame tileid="390" duration="100"/>
        <frame tileid="417" duration="100"/>
       </animation>
      </tile>
      <tile id="229">
       <animation>
        <frame tileid="229" duration="100"/>
        <frame tileid="256" duration="100"/>
        <frame tileid="283" duration="100"/>
        <frame tileid="310" duration="100"/>
        <frame tileid="337" duration="100"/>
        <frame tileid="364" duration="100"/>
        <frame tileid="391" duration="100"/>
        <frame tileid="418" duration="100"/>
       </animation>
      </tile>
      <tile id="230">
       <animation>
        <frame tileid="230" duration="100"/>
        <frame tileid="311" duration="100"/>
        <frame tileid="392" duration="100"/>
        <frame tileid="473" duration="100"/>
        <frame tileid="554" duration="100"/>
        <frame tileid="233" duration="100"/>
        <frame tileid="314" duration="100"/>
        <frame tileid="395" duration="100"/>
        <frame tileid="476" duration="100"/>
        <frame tileid="557" duration="100"/>
       </animation>
      </tile>
      <tile id="231">
       <animation>
        <frame tileid="231" duration="100"/>
        <frame tileid="312" duration="100"/>
        <frame tileid="393" duration="100"/>
        <frame tileid="474" duration="100"/>
        <frame tileid="555" duration="100"/>
        <frame tileid="234" duration="100"/>
        <frame tileid="315" duration="100"/>
        <frame tileid="396" duration="100"/>
        <frame tileid="477" duration="100"/>
        <frame tileid="558" duration="100"/>
       </animation>
      </tile>
      <tile id="232">
       <animation>
        <frame tileid="232" duration="100"/>
        <frame tileid="313" duration="100"/>
        <frame tileid="394" duration="100"/>
        <frame tileid="475" duration="100"/>
        <frame tileid="556" duration="100"/>
        <frame tileid="235" duration="100"/>
        <frame tileid="316" duration="100"/>
        <frame tileid="397" duration="100"/>
        <frame tileid="478" duration="100"/>
        <frame tileid="559" duration="100"/>
       </animation>
      </tile>
      <tile id="236">
       <animation>
        <frame tileid="236" duration="100"/>
        <frame tileid="238" duration="100"/>
        <frame tileid="240" duration="100"/>
        <frame tileid="291" duration="100"/>
        <frame tileid="293" duration="100"/>
        <frame tileid="344" duration="100"/>
        <frame tileid="346" duration="100"/>
        <frame tileid="348" duration="100"/>
        <frame tileid="399" duration="100"/>
        <frame tileid="401" duration="100"/>
       </animation>
      </tile>
      <tile id="237">
       <animation>
        <frame tileid="237" duration="100"/>
        <frame tileid="239" duration="100"/>
        <frame tileid="290" duration="100"/>
        <frame tileid="292" duration="100"/>
        <frame tileid="294" duration="100"/>
        <frame tileid="345" duration="100"/>
        <frame tileid="347" duration="100"/>
        <frame tileid="398" duration="100"/>
        <frame tileid="400" duration="100"/>
        <frame tileid="402" duration="100"/>
       </animation>
      </tile>
      <tile id="257">
       <animation>
        <frame tileid="257" duration="100"/>
        <frame tileid="338" duration="100"/>
        <frame tileid="419" duration="100"/>
        <frame tileid="500" duration="100"/>
        <frame tileid="581" duration="100"/>
        <frame tileid="260" duration="100"/>
        <frame tileid="341" duration="100"/>
        <frame tileid="422" duration="100"/>
        <frame tileid="503" duration="100"/>
        <frame tileid="584" duration="100"/>
       </animation>
      </tile>
      <tile id="259">
       <animation>
        <frame tileid="259" duration="100"/>
        <frame tileid="340" duration="100"/>
        <frame tileid="421" duration="100"/>
        <frame tileid="502" duration="100"/>
        <frame tileid="583" duration="100"/>
        <frame tileid="262" duration="100"/>
        <frame tileid="343" duration="100"/>
        <frame tileid="424" duration="100"/>
        <frame tileid="505" duration="100"/>
        <frame tileid="586" duration="100"/>
       </animation>
      </tile>
      <tile id="263">
       <animation>
        <frame tileid="263" duration="100"/>
        <frame tileid="265" duration="100"/>
        <frame tileid="267" duration="100"/>
        <frame tileid="318" duration="100"/>
        <frame tileid="320" duration="100"/>
        <frame tileid="371" duration="100"/>
        <frame tileid="373" duration="100"/>
        <frame tileid="375" duration="100"/>
        <frame tileid="426" duration="100"/>
        <frame tileid="428" duration="100"/>
       </animation>
      </tile>
      <tile id="264">
       <animation>
        <frame tileid="264" duration="100"/>
        <frame tileid="266" duration="100"/>
        <frame tileid="317" duration="100"/>
        <frame tileid="319" duration="100"/>
        <frame tileid="321" duration="100"/>
        <frame tileid="372" duration="100"/>
        <frame tileid="374" duration="100"/>
        <frame tileid="425" duration="100"/>
        <frame tileid="427" duration="100"/>
        <frame tileid="429" duration="100"/>
       </animation>
      </tile>
      <tile id="284">
       <animation>
        <frame tileid="284" duration="100"/>
        <frame tileid="365" duration="100"/>
        <frame tileid="446" duration="100"/>
        <frame tileid="527" duration="100"/>
        <frame tileid="608" duration="100"/>
        <frame tileid="287" duration="100"/>
        <frame tileid="368" duration="100"/>
        <frame tileid="449" duration="100"/>
        <frame tileid="530" duration="100"/>
        <frame tileid="611" duration="100"/>
       </animation>
      </tile>
      <tile id="285">
       <animation>
        <frame tileid="285" duration="100"/>
        <frame tileid="366" duration="100"/>
        <frame tileid="447" duration="100"/>
        <frame tileid="528" duration="100"/>
        <frame tileid="609" duration="100"/>
        <frame tileid="288" duration="100"/>
        <frame tileid="369" duration="100"/>
        <frame tileid="450" duration="100"/>
        <frame tileid="531" duration="100"/>
        <frame tileid="612" duration="100"/>
       </animation>
      </tile>
      <tile id="286">
       <animation>
        <frame tileid="286" duration="100"/>
        <frame tileid="367" duration="100"/>
        <frame tileid="448" duration="100"/>
        <frame tileid="529" duration="100"/>
        <frame tileid="610" duration="100"/>
        <frame tileid="289" duration="100"/>
        <frame tileid="370" duration="100"/>
        <frame tileid="451" duration="100"/>
        <frame tileid="532" duration="100"/>
        <frame tileid="613" duration="100"/>
       </animation>
      </tile>
      <tile id="438">
       <animation>
        <frame tileid="438" duration="100"/>
        <frame tileid="465" duration="100"/>
        <frame tileid="492" duration="100"/>
        <frame tileid="519" duration="100"/>
        <frame tileid="546" duration="100"/>
        <frame tileid="573" duration="100"/>
        <frame tileid="600" duration="100"/>
        <frame tileid="627" duration="100"/>
       </animation>
      </tile>
      <tile id="439">
       <animation>
        <frame tileid="439" duration="100"/>
        <frame tileid="466" duration="100"/>
        <frame tileid="493" duration="100"/>
        <frame tileid="520" duration="100"/>
        <frame tileid="547" duration="100"/>
        <frame tileid="574" duration="100"/>
        <frame tileid="601" duration="100"/>
        <frame tileid="628" duration="100"/>
       </animation>
      </tile>
      <tile id="440">
       <animation>
        <frame tileid="440" duration="100"/>
        <frame tileid="467" duration="100"/>
        <frame tileid="494" duration="100"/>
        <frame tileid="521" duration="100"/>
        <frame tileid="548" duration="100"/>
        <frame tileid="575" duration="100"/>
        <frame tileid="602" duration="100"/>
        <frame tileid="629" duration="100"/>
       </animation>
      </tile>
      <tile id="441">
       <animation>
        <frame tileid="441" duration="100"/>
        <frame tileid="468" duration="100"/>
        <frame tileid="495" duration="100"/>
        <frame tileid="522" duration="100"/>
        <frame tileid="549" duration="100"/>
        <frame tileid="576" duration="100"/>
        <frame tileid="603" duration="100"/>
        <frame tileid="630" duration="100"/>
       </animation>
      </tile>
      <tile id="442">
       <animation>
        <frame tileid="442" duration="100"/>
        <frame tileid="469" duration="100"/>
        <frame tileid="496" duration="100"/>
        <frame tileid="523" duration="100"/>
        <frame tileid="550" duration="100"/>
        <frame tileid="577" duration="100"/>
        <frame tileid="604" duration="100"/>
        <frame tileid="631" duration="100"/>
       </animation>
      </tile>
      <tile id="443">
       <animation>
        <frame tileid="443" duration="100"/>
        <frame tileid="470" duration="100"/>
        <frame tileid="497" duration="100"/>
        <frame tileid="524" duration="100"/>
        <frame tileid="551" duration="100"/>
        <frame tileid="578" duration="100"/>
        <frame tileid="605" duration="100"/>
        <frame tileid="632" duration="100"/>
       </animation>
      </tile>
      <tile id="444">
       <animation>
        <frame tileid="444" duration="100"/>
        <frame tileid="471" duration="100"/>
        <frame tileid="498" duration="100"/>
        <frame tileid="525" duration="100"/>
        <frame tileid="552" duration="100"/>
        <frame tileid="579" duration="100"/>
        <frame tileid="606" duration="100"/>
        <frame tileid="633" duration="100"/>
       </animation>
      </tile>
      <tile id="445">
       <animation>
        <frame tileid="445" duration="100"/>
        <frame tileid="472" duration="100"/>
        <frame tileid="499" duration="100"/>
        <frame tileid="526" duration="100"/>
        <frame tileid="553" duration="100"/>
        <frame tileid="580" duration="100"/>
        <frame tileid="607" duration="100"/>
        <frame tileid="634" duration="100"/>
       </animation>
      </tile>
      <tile id="452">
       <animation>
        <frame tileid="452" duration="100"/>
        <frame tileid="454" duration="100"/>
        <frame tileid="456" duration="100"/>
        <frame tileid="507" duration="100"/>
        <frame tileid="509" duration="100"/>
        <frame tileid="560" duration="100"/>
        <frame tileid="562" duration="100"/>
        <frame tileid="564" duration="100"/>
        <frame tileid="615" duration="100"/>
        <frame tileid="617" duration="100"/>
       </animation>
      </tile>
      <tile id="453">
       <animation>
        <frame tileid="453" duration="100"/>
        <frame tileid="455" duration="100"/>
        <frame tileid="506" duration="100"/>
        <frame tileid="508" duration="100"/>
        <frame tileid="510" duration="100"/>
        <frame tileid="561" duration="100"/>
        <frame tileid="563" duration="100"/>
        <frame tileid="614" duration="100"/>
        <frame tileid="616" duration="100"/>
        <frame tileid="618" duration="100"/>
       </animation>
      </tile>
      <tile id="479">
       <animation>
        <frame tileid="479" duration="100"/>
        <frame tileid="481" duration="100"/>
        <frame tileid="483" duration="100"/>
        <frame tileid="534" duration="100"/>
        <frame tileid="536" duration="100"/>
        <frame tileid="587" duration="100"/>
        <frame tileid="589" duration="100"/>
        <frame tileid="591" duration="100"/>
        <frame tileid="642" duration="100"/>
        <frame tileid="644" duration="100"/>
       </animation>
      </tile>
      <tile id="480">
       <animation>
        <frame tileid="480" duration="100"/>
        <frame tileid="533" duration="100"/>
        <frame tileid="535" duration="100"/>
        <frame tileid="537" duration="100"/>
        <frame tileid="588" duration="100"/>
        <frame tileid="590" duration="100"/>
        <frame tileid="641" duration="100"/>
        <frame tileid="643" duration="100"/>
        <frame tileid="645" duration="100"/>
       </animation>
      </tile>
     </tileset>
     <layer id="1" name="Base Layer" width="{mapw}" height="{mapw}">
      <data encoding="csv">""".format(mapw = str(MAP_SIZE), tile_count = str(TILE_COUNT + 228), tileset_columns = str(TILESET_COLUMNS), tilesheet = tile_set)
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

    # Makes and writes water layer
    print("Writing water layer...")
    next_layer_txt = """<layer id="3" name="Water" width="{mapw}" height="{mapw}">
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
    overlay2_list = np.ndarray.tolist(plant_layer_vals)
    next_layer_txt = """<layer id="4" name="Plants Waves Rivers" width="{mapw}" height="{mapw}">
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

    # Makes and writes tree layer
    print("Writing tree layer...")
    tree_list = np.ndarray.tolist(tree_layer_vals)
    next_layer_txt = """<layer id="5" name="Trees" width="{mapw}" height="{mapw}">
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
mask_slider = ravenui.Slider(ui, "Mask Intensity", (10, 10), 0.5, 1, 0, True)
ocean_slider = ravenui.Slider(ui, "Ocean Depth", (112, 10), OCEAN_FACTOR, 0.99, -1, True)
elevation_slider = ravenui.Slider(ui, "Max Elevation", (214, 10), SEALVL_FACTOR, 1.3, 0.0001, True)
swamps_slider = ravenui.Slider(ui, "Max Swamps", (316, 10), 10, 50, 0)
deserts_slider = ravenui.Slider(ui, "Max Deserts", (418, 10), 2, 20, 0)
river_slider = ravenui.Slider(ui, "River Sparsity", (520, 10), RIVER_CHANCE, 40, 5)
plant_slider = ravenui.Slider(ui, "Plant Density.", (622, 10), 1, 10, 0.1)
start_button = ravenui.Button(ui, "New Map", (10, 62), new_map, bg=(50, 200, 20))
update_button = ravenui.Button(ui, "Update Map", (10, 62), update_map, bg=(50, 200, 20))
update_button.hide()
tmx_button = ravenui.Button(ui, "Save TMX", (112, 62), make_tmx, bg=(50, 200, 20))
tmx_button.hide()
noise_button = ravenui.Button(ui, "Perlin Noise", (214, 62), switch_noise, bg=(50, 200, 20))
scale_slider = ravenui.Slider(ui, "Scale", (316, 62), 100, 200, 5, True)
octaves_slider = ravenui.Slider(ui, "Octaves", (418, 62), 6, 10, 1)
persistence_slider = ravenui.Slider(ui, "Persistence", (520, 62), 0.5, 1, 0.1, True)
lacunarity_slider = ravenui.Slider(ui, "Lacunarity", (622, 62), 2.0, 4, 0.1, True)

random_mask_button = ravenui.Button(ui, "New Mask", (622, 114), make_random_mask, bg=(50, 200, 20))
mnoise_button = ravenui.Button(ui, "Perlin Noise", (622, 166), switch_noise2, bg=(50, 200, 20))
mscale_slider = ravenui.Slider(ui, "Scale", (622, 218), 100, 400, 5, True)
moctaves_slider = ravenui.Slider(ui, "Octaves", (622, 270), 6, 10, 1)
mpersistence_slider = ravenui.Slider(ui, "Persistence", (622, 322), 0.5, 1, 0.1, True)
mlacunarity_slider = ravenui.Slider(ui, "Lacunarity", (622, 374), 2.0, 4, 0.1, True)
padding_slider = ravenui.Slider(ui, "Padding", (622, 422), 75, int(MAP_SIZE/5), 10)
max_islands_slider = ravenui.Slider(ui, "Max #Islands", (622, 474), 10, 100, 1)
min_island_size_slider = ravenui.Slider(ui, "Min Isize", (622, 526), 500, 10000, 20)
max_island_size_slider = ravenui.Slider(ui, "Max Isize", (622, 578), MAP_SIZE * 50, int(MAP_SIZE * MAP_SIZE * 0.8), 10000)
scope_slider = ravenui.Slider(ui, "Noise Depth", (622, 630), 0.6, 0.9, 0.1, True)
biome_button = ravenui.Button(ui, "Tropics", (622, 682), toggle_biome, bg=(50, 200, 20))
fromlist_button = ravenui.Button(ui, "Auto World", (622, 734), auto_world_map, bg=(50, 200, 20))

def draw():
    screen.fill(BLACK)
    screen.blit(scale_map_surface, (10, 115))
    ui.draw()
    pg.display.flip()

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
    if start_button.hidden:
        update_button.show()
    if True in [scale_slider.hit, persistence_slider.hit, lacunarity_slider.hit, octaves_slider.hit, noise_button.hit, random_mask_button.hit]:
        update_button.hide()
        start_button.show()
        tmx_button.hide()

    # Draw / render
    draw()
pg.quit()