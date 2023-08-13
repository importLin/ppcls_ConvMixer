def get_bpp(dat, pix_nums):
    return dat/pix_nums


dat = 1.5 * 1024 * 1024 * 1024 * 8
pix = 10000 * 224 * 224

print(get_bpp(dat, pix))