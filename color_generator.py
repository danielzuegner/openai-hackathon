import colorsys

N = 10 # number of colors

rgb_colors = [colorsys.hsv_to_rgb(i*1.0/N, 0.5, 0.5) for i in range(N)]

print(rgb_colors)