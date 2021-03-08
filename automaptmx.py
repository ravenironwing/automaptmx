#import matplotlib.pyplot as plt
import ravenui
import numpy as np
import pygame as pg
import noise, cv2 # pip3 install opencv-python
import random
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
TILE_COUNT = 324
TILESET_COLUMNS = 27
SHALLOW_TILE = 217
WATER_TILE = 244
SWAMP_SHALLOWS = 271
SWAMP_WATER = 298
MAX_SWAMP_SIZE = 100 * 100 # Max area swamps can take up.
KINDSOFTILES = 18 #Ignoring the water tiles and swamp tiles
ROUND_TILES = range(89, 133)
ROUNDPOINT_TILES = range(155, 177)
MASK_FACTOR = 0
ELEVATION_FACTOR = 0
SEALVL_FACTOR = 1

base_layer_vals = []
water_layer_vals = []
BLACK = [0, 0, 0]
colors = [[93, 120, 150], [108, 135, 168], [147, 174, 171], [165, 195, 228], [189, 219, 219], [261, 231, 174], [152, 118, 84], [102, 159, 69], [75, 141, 33], [102, 66, 45], [120, 105, 72], [123, 108, 84], [144, 144, 144], [168, 168, 168], [183, 183, 183], [225, 225, 243], [255, 255, 255], [240, 250, 255]]
swampcolors = [[0, 155, 130], [0, 175, 110], [0, 195, 90], [0, 210, 70], [0, 215, 50], [0, 220, 15], [0, 230, 10], [0, 240, 5], [0, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

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
world = np.zeros((MAP_SIZE, MAP_SIZE))
swamps_arr = np.zeros((MAP_SIZE, MAP_SIZE), int) # Used for storing regions that can be turned into swamps.
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
    return color_world

def new_map():
    global scale_map_surface, world_noise, world, mask, swamps_mask
    "Building new map..."
    # Noise settings
    # Creates a perlin noise array the size of the map.
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
    max_grad = np.amax(world)
    world = world / max_grad

    """
    swamps_mask = np.zeros((MAP_SIZE, MAP_SIZE))
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            swamps_mask[i][j] = noise.snoise2(i/scale_slider.val,
                                        j/scale_slider.val,
                                        octaves=octaves_slider.val,
                                        persistence=persistence_slider.val,
                                        lacunarity=lacunarity_slider.val,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=0) + 0.5  # The 0.5 makes i return values between 0 and 1.
    max_grad = np.amax(world)
    world = world / max_grad
    max_grad = np.amax(swamps_mask)
    swamps_mask = swamps_mask / max_grad
    world_noise = np.zeros_like(world)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            world_noise[i][j] = ((world[i][j]+elev_slider.val) * mask[i][j])
            #if world_noise[i][j] > 0:
            #    world_noise[i][j] *= 20
    print(np.max(world_noise))
    print(np.min(world_noise))
    # get it between 0 and 1
    max_grad = np.amax(world_noise)
    world_noise = world_noise / max_grad

    continent_grad = add_color(world_noise)
    new_continent = np.uint8(continent_grad)
    #im = Image.fromarray(new_continent)
    #im.save("continent.png")
    pg.surfarray.blit_array(map_surface, new_continent)
    temp_surf = pg.transform.rotate(map_surface, -90)
    temp_surf2 = pg.transform.flip(temp_surf, True, False)
    scale_map_surface = pg.transform.scale(temp_surf2, (500, 500))"""
    update_map()

def update_map():
    global world_noise, scale_map_surface, world, mask, swamps_arr, MAX_SWAMP_SIZE
    swamps_arr = np.zeros((MAP_SIZE, MAP_SIZE), int)
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
                if world_noise[i][j] > 0.99:
                    world_noise[i][j] = 0.99
                if world_noise[i][j] < 0.01:
                    world_noise[i][j] = 0.01

    # Figures out regions where swamps can be and stores each swamp region in a list.
    temp_array = np.array(world_noise)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            x = temp_array[i][j]
            newx = int(x * (KINDSOFTILES - 1))
            newx += 1  # Gets rid of zeros
            temp_array[i][j] = newx
    grass_arr = np.where(temp_array < 9, 255, 0)
    grass_img = grass_arr.astype('uint8')
    kernel = np.ones((3, 3), np.uint8)
    grass_img = cv2.dilate(grass_img, kernel, iterations=1) # This adds a padding of 1 around the swamp areas so the edges are correct.
    shade = 0
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if grass_img[x, y] == 255:
                shade += 1
                if shade > 255: # Limited to 255 swamps
                    shade = 255
                    break
                cv2.floodFill(grass_img, None, (y, x), shade)
    print("Potential swamp areas: ", shade)
    swamps_list = []
    for color in range(0, shade):
        swamp_arr = np.where(grass_img == color, 1, 0)
        if np.count_nonzero(swamp_arr == 1) < MAX_SWAMP_SIZE:
            swamps_list.append(swamp_arr)
    print("Correct size swamp areas:" + str(len(swamps_list)))
    if len(swamps_list) < swamps_slider.val:
        num_swamps = len(swamps_list)
    else:
        num_swamps = swamps_slider.val
    swamps_list = random.sample(swamps_list, num_swamps) # Reduces number of swamps in list to up to the selected number.
    for swamp in swamps_list:
        swamps_arr = np.add(swamps_arr, swamp)
    
    """
    # Combines swamp noise with world noise then scales it.
    swamps_noise = np.zeros_like(world)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            swamps_noise[i][j] = ((world_noise[i][j]) * (swamps_mask[i][j]))
    max_grad = np.amax(swamps_noise)
    swamps_noise = swamps_noise / max_grad"""

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
    print("Saving tmx file...")
    max_num = 0
    min_num = 20
    #Makes base layer
    for i in range(MAP_SIZE):
        new_base_row = []
        new_water_row = []
        for j in range(MAP_SIZE):
            x = world_noise[i][j]
            newx = int(x * (KINDSOFTILES-1))
            newx += 1  # Gets rid of zeros
            if swamps_arr[i][j] == 1: # Swiches tiles to swamp tiles for swamp areas.
                newx = newx + KINDSOFTILES #Switches to swamp tileset columns
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
            elif newx == 5:
                water = SHALLOW_TILE
            elif newx < 5:
                water = WATER_TILE
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
    # Creates a temporary array with a padding of 1 to search for replace pattern
    base_arr = np.array(base_layer_vals, int)
    y_size, x_size = base_arr.shape
    arr1 = np.zeros((y_size + 2, x_size + 2), int)
    arr1[1:-1, 1:-1] = base_arr
    for x in range(1, x_size + 1):      # In this code I'm finding tile patterns by counting tiles of neighboring types in 3x3 and 2x2 groups instead of checking each tile in the group.
        print("Column " + str(x) + " out of " + str(MAP_SIZE) + ".")
        for y in range(1, y_size + 1):
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

            for tile in range(TILESET_COLUMNS, 1, -1):
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

                if (num_tiles == 1) and (num_under_tiles == 3):
                    # rounds out positive corners
                    if arr1[x][y] == tile:
                        corners_layer_vals[x - 1][y - 1] = ul_corner(tile)
                    elif arr1[x+1][y] == tile:
                        corners_layer_vals[x][y - 1] = ur_corner(tile)
                    elif arr1[x][y+1] == tile:
                        corners_layer_vals[x - 1][y] = ll_corner(tile)
                    elif arr1[x+1][y+1] == tile:
                        corners_layer_vals[x][y] = lr_corner(tile)

                if (num_hv_tiles == 4) and (tile - 1 == arr1[x][y]): # Fixes ugly single holes
                    if tile in [6, 27]:
                        overlay2_layer_vals[x -1][y - 1] = tile + (4 * TILESET_COLUMNS) # Sand water pools in water layer instead of corners layer.
                        break
                    else:
                        corners_layer_vals[x -1][y - 1] = tile + (4 * TILESET_COLUMNS)
                        break
                elif (num_hv_under_tiles == 4) and (tile == arr1[x][y]): # Fixes ugly single tile
                    corners_layer_vals[x -1][y - 1] = tile + (5 * TILESET_COLUMNS)
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
                        break
                    elif arr1[x+1][y] == tile and (arr1[x-1][y - 1] != tile) and (arr1[x-1][y + 1] != tile):
                        corners_layer_vals[x - 1][y - 1] = pr_corner(tile)
                        break
                    elif arr1[x][y+1] == tile and (arr1[x-1][y - 1] != tile) and (arr1[x+1][y - 1] != tile):
                        corners_layer_vals[x - 1][y - 1] = pd_corner(tile)
                        break
                    elif arr1[x][y-1] == tile and (arr1[x-1][y + 1] != tile) and (arr1[x+1][y + 1] != tile): #left
                        corners_layer_vals[x - 1][y - 1] = pu_corner(tile)
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

    """
    # Adds swamp tiles
    temp_array = np.array(base_layer_vals, int)
    grass_arr = np.where(temp_array < 9, 255, 0)
    grass_img = grass_arr.astype('uint8')
    #grass_img = cv2.cvtColor(grass_arr, cv2.COLOR_GRAY2BGR)
    shade = 0
    for x in range(MAP_SIZE):
        print(x)
        for y in range(MAP_SIZE):
            if grass_img[x, y] == 255:
                shade += 1
                if shade > 255: # Limited to 255 swamps
                    shade = 255
                    break
                cv2.floodFill(grass_img, None, (y, x), shade)
    print("Number of elements: ", shade)
    
    swamps_list = [] # Creates a list of potential swamp areas.
    for color in range(0, shade):
        swamp_arr = np.where(temp_array == color, 1, 0)
        swamps_list.append(swamp_arr)

    swamp_factor = 13 # Collums from the sand to swamp sand tiles.
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if base_layer_vals[i][j] in range(6, 10) and (swamps_regions[i][j] != 0):
                if swamps_regions[i][j] == 1:
                    base_layer_vals[i][j] += swamp_factor  # Shifts to swamp tile columns.
                if corners_layer_vals[i][j] != 0:
                    corners_layer_vals[i][j] = (corners_layer_vals[i][j] + swamp_factor) | return_flags(corners_layer_vals[i][j])
                if overlay_layer_vals[i][j] != 0:
                    overlay_layer_vals[i][j] = (overlay_layer_vals[i][j] + swamp_factor) | return_flags(overlay_layer_vals[i][j])
                if overlay2_layer_vals[i][j] != 0:
                    overlay2_layer_vals[i][j] = (overlay2_layer_vals[i][j] + swamp_factor) | return_flags(overlay2_layer_vals[i][j])
                if base_layer_vals[i][j] in [19, 21]: # Adds shallows swamp water.
                    water_layer_vals[i][j] += SWAMP_SHALLOWS
                if base_layer_vals[i][j] == 20: # Adds swamp water.
                    water_layer_vals[i][j] += SWAMP_WATER"""

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
      <data encoding="csv">""".format(mapw = str(MAP_SIZE), watertile = str(WATER_TILE), tile_count = str(TILE_COUNT), tileset_columns = str(TILESET_COLUMNS))
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
elev_slider = ravenui.Slider(ui, "Land Density", (112, 10), ELEVATION_FACTOR, 1, -1, True)
sealvl_slider = ravenui.Slider(ui, "Elevation", (214, 10), SEALVL_FACTOR, 2, 0.0001, True)
scale_slider = ravenui.Slider(ui, "Scale", (316, 10), 100, 200, 5, True)
octaves_slider = ravenui.Slider(ui, "Octaves", (418, 10), 6, 10, 1)
persistence_slider = ravenui.Slider(ui, "Persistence", (520, 10), 0.5, 1, 0.1, True)
lacunarity_slider = ravenui.Slider(ui, "Lacunarity", (520, 10), 2.0, 4, 0.1, True)
swamps_slider = ravenui.Slider(ui, "Max Swamps", (622, 10), 10, 100, 0)
start_button = ravenui.Button(ui, "New Map", (10, 62), new_map, bg=(50, 200, 20))
update_button = ravenui.Button(ui, "Update Map", (112, 62), update_map, bg=(50, 200, 20))
#update_button.hide()
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