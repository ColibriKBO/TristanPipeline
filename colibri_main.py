"""August 2019 edit"""

import sep
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.convolution import convolve_fft, MexicanHat1DKernel
from astropy.time import Time
from joblib import delayed, Parallel
from copy import deepcopy
import multiprocessing
import time
import datetime
import numpy.ma as ma
import os
import struct
import gc
#import wcsget  # adaptation of Astrometry.net API


def getNum(f):
    ''''""" Extract Unix time from .vid filename """

    num = f.split('_')[1]
    return int(num, 16)'''
    num = f.split('/')[2].split('ms')[0]
    return int(num, 16)


def diffMatch(template, data, unc):
    """ Calculates the best start position (minX) and the minimization constant associated with this match (minChi) """

    minChi = np.inf
    minX = np.inf
    for st in xrange(((len(data) / 2) - len(template) / 2) - 3, (len(data) / 2) - (len(template) / 2) + 3):
        sum = 0
        for val in xrange(0, len(template)):
            sum += (abs(template[val] - data[st + val])) / abs(unc[st + val])
        if sum < minChi:
            minChi = sum
            minX = st
    return minChi, minX


'''def initialFind(data, flat, dark):
    """ Locates the stars in the initial time slice """

    """ Background extraction for initial time slice"""
    data_new = deepcopy(data)
    data_new[1:, :] /= flat[1:, :]
    data_new -= dark
    bkg = sep.Background(data_new)
    bkg.subfrom(data_new)
    thresh = 3. * bkg.globalrms  # set detection threshold to mean + 3 sigma
    """ Identify stars in initial time slice """
    objects = sep.extract(data_new, thresh)

    """ Characterize light profile of each star"""
    halfLightRad = np.sqrt(objects['npix'] / np.pi) / 2.  # approximate half light radius as half of radius

    """ Generate tuple of (x,y) positions for each star"""
    positions = zip(objects['x'], objects['y'])

    return positions, halfLightRad, thresh'''


def initialFindFITS(data):
    """ Locates the stars in the initial time slice """

    """ Background extraction for initial time slice"""
    data_new = deepcopy(data)
    bkg = sep.Background(data_new)
    bkg.subfrom(data_new)
    thresh = 3. * bkg.globalrms  # set detection threshold to mean + 3 sigma

    """ Identify stars in initial time slice """
    objects = sep.extract(data_new, thresh)

    """ Characterize light profile of each star"""
    halfLightRad = np.sqrt(objects['npix'] / np.pi) / 2.  # approximate half light radius as half of radius

    """ Generate tuple of (x,y,r) positions for each star"""
    positions = zip(objects['x'], objects['y'], halfLightRad)

    return positions


def refineCentroid(data, coords, sigma):
    """ Refines the centroid for each star for a set of test slices of the data cube """

    x_initial = [pos[0] for pos in coords]
    y_initial = [pos[1] for pos in coords]
    new_pos = np.array(sep.winpos(data, x_initial, y_initial, sigma, subpix=5))[0:2, :]
    x = new_pos[:][0].tolist()
    y = new_pos[:][1].tolist()
    return zip(x, y)


def averageDrift(positions, end, frames):
    """ Determines the per-frame drift rate of each star """

    x_drift = np.array([np.subtract(positions[t, :, 0], positions[t - 1, :, 0]) for t in range(1, end)])
    y_drift = np.array([np.subtract(positions[t, :, 1], positions[t - 1, :, 1]) for t in range(1, end)])
    return np.median(x_drift[1:], 0) / frames, np.median(y_drift[1:], 0) / frames


'''def timeEvolve(data, coords, x_drift, y_drift, r, stars, x_length, y_length, flat, dark, t):
    """ Adjusts aperture based on star drift and calculates flux in aperture"""

    x = [coords[ind, 0] + x_drift[ind] for ind in range(0, stars)]
    y = [coords[ind, 1] + y_drift[ind] for ind in range(0, stars)]
    inds = clipCutStars(x, y, x_length, y_length)
    inds = list(set(inds))
    inds.sort()
    xClip = np.delete(np.array(x), inds)
    yClip = np.delete(np.array(y), inds)
    data[1:, :] /= flat[1:, :]
    data -= dark
    bkg = np.median(data) * np.pi * r * r
    fluxes = (sep.sum_circle(data, xClip, yClip, r)[0] - bkg).tolist()
    for a in inds:
        fluxes.insert(a, 0)
    coords = zip(x, y, fluxes, [t] * len(x))
    return coords'''


def timeEvolveFITS(data, coords, x_drift, y_drift, r, stars, x_length, y_length, t):
    """ Adjusts aperture based on star drift and calculates flux in aperture"""

    x = [coords[ind, 0] + x_drift[ind] for ind in range(0, stars)]
    y = [coords[ind, 1] + y_drift[ind] for ind in range(0, stars)]
    inds = clipCutStars(x, y, x_length, y_length)
    inds = list(set(inds))
    inds.sort()
    xClip = np.delete(np.array(x), inds)
    yClip = np.delete(np.array(y), inds)
    bkg = np.median(data) * np.pi * r * r
    fluxes = (sep.sum_circle(data, xClip, yClip, r)[0] - bkg).tolist()
    for a in inds:
        fluxes.insert(a, 0)
    coords = zip(x, y, fluxes, [t] * len(x))
    return coords


def timeEvolveFITSNoDrift(data, coords, x_drift, y_drift, r, stars, x_length, y_length, t):
    x = [coords[ind, 0] for ind in range(0, stars)]
    y = [coords[ind, 1] for ind in range(0, stars)]
    bkg = np.median(data) * np.pi * r * r
    fluxes = (sep.sum_circle(data, x, y, r)[0] - bkg).tolist()
    coords = zip(x, y, fluxes, [t] * len(x))
    return coords


def clipCutStars(x, y, x_length, y_length):
    """ When the aperture is near the edge of the field of view, sets flux to zero to prevent fadeout"""

    r = 20.
    xeff = np.array(x)
    yeff = np.array(y)
    ind = np.where(r > xeff)
    ind = np.append(ind, np.where(xeff >= (x_length - r)))
    ind = np.append(ind, np.where(r > yeff))
    ind = np.append(ind, np.where(yeff >= (y_length - r)))
    return ind


def kernelDetection(fluxProfile, kernel, kernels, num):
    """ Detects dimming using Mexican Hat kernel for dip detection and set of Fresnel kernels for kernel matching """

    """ Prunes profiles"""
    light_curve = np.trim_zeros(fluxProfile[1:])
    if len(light_curve) == 0:
        return -2  # reject empty profiles
    med = np.median(light_curve)
    indices = np.where(light_curve > min(med * 2, med + 5 * np.std(light_curve)))
    light_curve = np.delete(light_curve, indices)
    trunc_profile = np.where(light_curve < 0, 0, light_curve)
    if len(trunc_profile) < 1100:
        return -2  # reject objects that go out of frame rapidly, ensuring adequate evaluation of median flux
    if abs(np.mean(trunc_profile[:1000]) - np.mean(trunc_profile[-1000:])) > np.std(trunc_profile[:1000]):
        return -2  # reject tracking failures
    if np.median(trunc_profile) < 5000:
        return -2  # reject stars that are very dim, as SNR is too poor
    m = np.std(trunc_profile[900:1100]) / np.median(trunc_profile[900:1100])



    """ Dip detection"""
    # astropy v2.0+ changed convolve_fft quite a bit... see documentation, for now normalize_kernel=False
    conv = convolve_fft(trunc_profile, kernel, normalize_kernel=False)
    minPeak = np.argmin(conv)
    minVal = min(conv)

    # if geometric dip (greater than 40%), flag as candidate without template matching
    norm_trunc_profile = trunc_profile/np.median(trunc_profile)
    if norm_trunc_profile[minPeak] < 0.6:
        critTime = np.where(fluxProfile == light_curve[minPeak])[0]
        print datetime.datetime.now(), "Detected >40% dip: frame", str(critTime) + ", star", num
        return critTime[0]

    if 30 <= minPeak < len(trunc_profile) - 30:
        med = np.median(
            np.concatenate((trunc_profile[minPeak - 100:minPeak - 30], trunc_profile[minPeak + 30:minPeak + 100])))
        trunc_profile = trunc_profile[(minPeak - 30):(minPeak + 30)]
        unc = ((np.sqrt(abs(trunc_profile) / 100.) / np.median(trunc_profile)) * 100)
        unc = np.where(unc == 0, 0.01, unc)
        trunc_profile /= med
    else:
        return -2  # reject events that are cut off at the start/end of time series
    bkgZone = conv[10: -10]

    if minVal < np.mean(bkgZone) - 3.75 * np.std(bkgZone):  # dip detection threshold
        """ Kernel matching"""
        old_min = np.inf
        for ind in range(0, len(kernels)):
            if min(kernels[ind]) > 0.8:
                continue
            new_min, loc = diffMatch(kernels[ind], trunc_profile, unc)
            if new_min < old_min:
                active_kernel = ind
                min_loc = loc
                old_min = new_min
        unc_l = unc[min_loc:min_loc + len(kernels[active_kernel])]
        thresh = 0
        for u in unc_l:
            thresh += (abs(m) ** 1) / (abs(u) ** 1)  # kernel match threshold
        if old_min < thresh:
            critTime = np.where(fluxProfile == light_curve[minPeak])[0]
            print datetime.datetime.now(), "Detected candidate: frame", str(critTime) + ", star", num
            if len(critTime) > 1:
                raise ValueError
            return critTime[0]  # returns location in original time series where dip occurs
        else:
            return -1  # reject events that do not pass kernel matching
    else:
        return -1  # reject events that do not pass dip detection


def readByte(file, start, length):
    """ Returns the integer located at a given position in the file """

    file.seek(start)
    byte = file.read(length)
    return struct.unpack('i', byte + (b'\0' * (4 - len(byte))))[0]


'''def getSize(filename, iterations):
    """ Imports file from .vid format """

    with open(filename, "rb") as file:
        magic = readByte(file, 0, 4)
        if magic != 809789782:
            print datetime.datetime.now(), "Error: invalid .vid file"
            return
        totBytes = os.stat(filename).st_size  # size of file in bytes
        seqlen = readByte(file, 4, 4)  # size of image data in bytes
        width = readByte(file, 30, 2)  # image width
        height = readByte(file, 32, 2)  # image height
        frames = totBytes // seqlen  # number of frames in the image
        bytesToNum = readByte(file, 34, 2) // 8  # number of bytes per piece of data
        if frames % 2 == 0:
            priorByte = totBytes / 2 * iterations
            frames /= 2
        else:
            if iterations == 0:
                priorByte = 0
                frames = (frames - 1) / 2
            else:
                frames = (frames + 1) / 2
                priorByte = (frames - 1) * bytesToNum

        file.seek(priorByte, os.SEEK_SET)
        x = np.fromfile(file, dtype='int32', count=width * height * frames / 2)
        unixTime = x[5::width * height / 2]  # get seconds since epoch  # TODO: only works when .2 --> 2 (integer)
        micro = x[6::width * height / 2] / 1000000.  # get additional milliseconds
        time_list = [np.float(z) + np.float(y) for y, z in zip(unixTime, micro)]

    return width, height, frames, time_list'''


def getSizeFITS(filenames):
    """ gets dimensions of fits 'video' """
    filename_first = filenames[0]
    filename_last = filenames[-1]
    frames = len(filenames)

    file = fits.open(filename_first)
    header = file[0].header
    width = header['NAXIS1']
    height = header['NAXIS1']

    time_start = Time(header['DATE-OBS'], format='fits', precision=9).unix

    file = fits.open(filename_last)
    header = file[0].header
    time_end = Time(header['DATE-OBS'], format='fits', precision=9).unix

    time_list = np.linspace(time_start, ime_end, frames)

    return width, height, frames, time_list


'''def importFrames(filename, iterations, frameNum, length):
    """ Imports file from .vid format """

    with open(filename, "rb") as file:
        magic = readByte(file, 0, 4)
        if magic != 809789782:
            print datetime.datetime.now(), "Error: invalid .vid file"
            return
        totBytes = os.stat(filename).st_size  # size of file in bytes
        seqlen = readByte(file, 4, 4)  # size of image data in bytes
        headlen = readByte(file, 8, 4)  # size of header data in bytes
        width = readByte(file, 30, 2)  # image width
        height = readByte(file, 32, 2)  # image height
        bytesToNum = readByte(file, 34, 2) // 8  # number of bytes per piece of data
        frames = totBytes // seqlen  # number of frames in the image
        area = width * height
        headerData = headlen // bytesToNum
        if frames % 2 == 0:
            priorByte = totBytes / 2 * iterations
        else:
            if iterations == 0:
                priorByte = 0
            else:
                frames = (frames + 1) / 2
                priorByte = (frames - 1) * bytesToNum
        file.seek(priorByte + width * height * bytesToNum * frameNum, os.SEEK_SET)
        c = width * height * length
        if c < 0:
            c = -1
        x = np.fromfile(file, dtype='uint16', count=c)
        for headerCell in range(0, headerData):
            x[headerCell::area] = 0
    x = np.reshape(x, [-1, width, height])
    if x.shape[0] == 1:
        x = x[0]
        x = x.astype('float64')
    return x'''


def importFramesFITS(filenames, frame_num, length):
    """ reads in frames from fits files """
    x = []
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= frame_num and i < frame_num + length]
    for filename in files_to_read:
        with open(filename, "rb") as file:
            data = fits.getdata(filename)
            x.append(data)
    x = np.array(x, dtype='float64')
    if x.shape[0] == 1:
        x = x[0]
        x = x.astype('float64')
    return x


def camCompare(ind, results, positions, name_stamp, day_stamp, directory):
    """ Saves data for later comparison between EMCCD1 and EMCCD2 using compareStreams.py """

    ''' select only the xy coordinates of time series flagged as containing occultation event '''
    flaggedPos = positions[:, ind, :]
    shp = flaggedPos[:, :, 0].shape[:2]
    xs = flaggedPos[0, :, 0]
    ys = flaggedPos[0, :, 1]

    ''' convert xy coordinates to FITS binary table format'''
    c1 = fits.Column(name='XIMAGE', format='D', array=xs)
    c2 = fits.Column(name='YIMAGE', format='D', array=ys)
    tbhdu = fits.BinTableHDU.from_columns([c1, c2])
    tbhdu.writeto("xycoords.xyls", clobber=True)

    ''' convert xy coordinates to RA dec using command line astrometry.net package '''
    os.system('wcs-xy2rd -i xycoords.xyls -o rdcoords.rdls -X XIMAGE -Y YIMAGE -w wcs.fits')

    '''convert resulting rdls file to slices in numpy ndarray'''
    hdulist = fits.open('rdcoords.rdls')
    rds = np.array(hdulist[1].data, dtype=tuple)
    xs = np.array([val[0] for val in rds], dtype=np.float64)
    ys = np.array([val[1] for val in rds], dtype=np.float64)
    xs = np.repeat(xs, shp[0])
    ys = np.repeat(ys, shp[0])
    xs = np.reshape(xs, shp)
    ys = np.reshape(ys, shp)
    flaggedPos = np.dstack((xs, ys, flaggedPos[:, :, 0], flaggedPos[:, :, 1], flaggedPos[:, :, 2], flaggedPos[:, :, 3]))

    ''' shorten time series to focus around occultation event '''
    centres = results[np.where(results > 0)]
    centredPos = np.empty([60, len(centres)], dtype=(np.float64, 6))
    for n in range(0, len(centres)):
        centredPos[:, n] = flaggedPos[centres[n] - 30:centres[n] + 30, n, :]
    np.save(str(day_stamp) + "/" + name_stamp + ".npy", centredPos)  # save x,y,RA,dec,flux,time series
    dt = np.dtype(
        [('starID', np.str, 50), ('RA', np.float64), ('dec', np.float64), ('fluxes', tuple), ('time', np.float64)])
    savedVals = np.empty([len(ind)], dtype=dt)
    for v in range(0, len(ind)):
        savedVals[v] = (
        name_stamp + "-" + str(ind[v]), centredPos[0, v, 0], centredPos[0, v, 1], tuple(centredPos[:, v, 4]),
        centredPos[30, v, 5])

    ''' save data for each camera for later comparison '''
    if directory == '/emccd1/':
        try:
            stream1 = np.load('temp1.npy')
            stream1 = np.append(stream1, savedVals)
        except IOError:
            stream1 = savedVals
        np.save('temp1.npy', stream1)  # if first camera, save matches
        print datetime.datetime.now(), "Directory:", directory
    elif directory == '/emccd2/':
        try:
            stream2 = np.load('temp2.npy')
            stream2 = np.append(stream2, savedVals)
        except IOError:
            stream2 = savedVals
        np.save('temp2.npy', stream2)  # if first camera, save matches
        print datetime.datetime.now(), "Directory:", directory
    else:
        print "Warning: directory not found"


def main(file, field_name):
    """ Detect possible occultation events in selected file and archive results """

    print datetime.datetime.now(), "Opening:", file

    """ create folder for results """
    day_stamp = datetime.date.today()
    if not os.path.exists(str(day_stamp)):
        os.makedirs(str(day_stamp))

    """ adjustable parameters """
    r = 5.  # radius of aperture for flux measurements
    expected_length = 0.15  #  related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing
    refresh_rate = 2.  # number of seconds (as float) between centroid refinements

    """ get file shape """
    filenames = glob(file + '*.fit')
    filenames.sort()

    # get shape of data, including list of unix times
    x_length, y_length, t_length, time_list = getSizeFITS(filenames) #t_length, time_list = getSize(filenames[0], half)
    print datetime.datetime.now(), "Imported", t_length, "frames"
    if t_length < 500:
        print datetime.datetime.now(), "Insufficient length data cube, skipping..."
        return

    """ prepare kernels """
    exposure_time = 0.025  # exposure length in seconds
    kernel_set = np.loadtxt('kernels25test.txt')

    kernel_frames = int(round(expected_length / exposure_time))
    mex_hat_kernel = MexicanHat1DKernel(kernel_frames)
    evolution_frames = int(round(refresh_rate / exposure_time))  # determines the number of frames in X seconds of data

    """ load/create positional data"""
    first_frame = importFramesFITS(filenames, 0, 1)
    field_path = str(day_stamp) + '/' + field_name + '_pos.npy'

    # if no positional data for current field, create it from first_frame
    if not os.path.exists(field_path):
        print datetime.datetime.now(), field_name, 'starfinding...',
        star_find_results = initialFindFITS(first_frame)
        np.save(field_path, star_find_results)
        print 'done'

    initial_positions = np.load(field_path)
    radii = initial_positions[:,-1]
    initial_positions = initial_positions[:,:-1]

    star_num = len(initial_positions) # number of stars in image

    """ centroid refinements and drift check """
    drift = False
    check_frames = t_length // evolution_frames
    test_pos = np.empty([check_frames, star_num], dtype=(np.float64, 2))
    radius = np.mean(radii * 2. / 2.35)  # calculate gaussian sigma for each star's light profile

    # refine centroids for drift measurements
    test_pos[0] = refineCentroid(first_frame, initial_positions, radius)
    for t in range(1, check_frames):
        test_pos[t] = refineCentroid(importFramesFITS(filenames, t * evolution_frames, 1), deepcopy(test_pos[t - 1]), radius)

    # check drift rates
    x_drift, y_drift = averageDrift(test_pos, check_frames, evolution_frames)  # drift rates per frame
    if abs(np.median(x_drift)) > 1e-2 or abs(np.median(y_drift)) > 1e-2:
        drift = True
        #if abs(np.median(x_drift)) > 1 or abs(np.median(y_drift)) > 1:
        #    print datetime.datetime.now(), "Significant drift, skipping..."  # find how much drift is too much
        #    return

    """ flux and time calculations with optional time evolution """
    data = np.empty([t_length, star_num], dtype=(np.float64, 4))
    bkg_first = np.median(first_frame) * np.pi * r * r
    data[0] = zip(initial_positions[:,0], initial_positions[:,1], (sep.sum_circle(first_frame, initial_positions[:,0], initial_positions[:,1], r)[0] - bkg_first).tolist(), np.ones(np.shape(np.array(initial_positions))[0]) * time_list[0])

    if drift:  # time evolve moving stars
        for t in range(1, t_length):
            data[t] = timeEvolveFITS(importFramesFITS(filenames, t, 1), deepcopy(data[t - 1]), x_drift, y_drift,
                                     r, star_num, x_length, y_length, flat, dark, time_list[t])
    else:  # if there is not significant drift, approximate drift to 0
        for t in range(1, t_length):
            x_drift, y_drift = np.empty(np.shape(x_drift)), np.empty(np.shape(y_drift))
            data[t] = timeEvolveFITSNoDrift(importFramesFITS(filenames, t, 1), deepcopy(data[t - 1]), x_drift, y_drift,
                                     r, star_num, x_length, y_length, time_list[t])

    # plotting star locations in first frame - for debugging
    '''from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots()
    fradat = importFramesFITS(filenames, half, 0, 1)
    m, s = np.mean(fradat), np.std(fradat)
    im = ax.imshow(fradat, interpolation='nearest', cmap='gray', vmin=m - s, vmax=m + 5 * s, origin='lower')
            
    for i in range(star_num):
        l = Circle(xy=(data[0, i, 0], data[t, i, 1]), radius=5 / 2.)
        l.set_facecolor('none')
        l.set_edgecolor('green')
        ax.add_artist(l)
    plt.show()'''

    print datetime.datetime.now(), 'Photometry done', data.shape  #positions.shape
    # data is an array of shape: [frames, star_num, {0:x, 1:y, 2:flux, 3:unix_time}]

    """ testing -> remove this """
    k_test = np.loadtxt('kernel_test.txt')
    k_test = np.loadtxt('kernel_dip_test.txt')
    test_star = 0
    data[250:250 + len(k_test), test_star, 2] *= k_test


    """ kernel matching """
    cores = multiprocessing.cpu_count()  # determine number of CPUs for parallel processing
    results = np.array(Parallel(n_jobs=cores, backend='threading')(
        delayed(kernelDetection)(data[:, index, 2], mex_hat_kernel, kernel_set, index) for index in
        range(0, star_num)))  # perform dip detection and kernel match for all time series

    """ data archival """
    name_stamp = str(file.split('/')[-2])
    save_times = results[np.where(results > 0)]  # frames with kernel matches for each star
    save_chunk = int(round(5 / exposure_time))  # save 10 seconds of frames around event

    for t in save_times:  # save data surrounding candidate event
        if t - save_chunk <= 0:  # if chunk includes lower data boundary, start at 0
            np.save("./" + str(day_stamp) + "/Surrounding-" + name_stamp + "-" + str(np.where(results == t)[0][0]) + ".npy", importFramesFITS(filenames, 0, t + save_chunk))
            np.save("./" + str(day_stamp) + "/timing-" + name_stamp + "-" + str(np.where(results == t)[0][0]) + ".npy", time_list[:t + save_chunk])
        else:  # if chunk does not include lower data boundary
            if t + save_chunk >= t_length:  # if chunk includes upper data boundary, stop at upper boundary
                np.save("./" + str(day_stamp) + "/Surrounding-" + name_stamp + "-" + str(np.where(results == t)[0][0]) + ".npy", importFramesFITS(filenames, t - save_chunk, t_length - t + save_chunk))
                np.save("./" + str(day_stamp) + "/timing-" + name_stamp + "-" + str(np.where(results == t)[0][0]) + ".npy", time_list[t - save_chunk:])
            else:  # if chunk does not include upper data boundary
                np.save("./" + str(day_stamp) + "/Surrounding-" + name_stamp + "-" + str(np.where(results == t)[0][0]) + ".npy", importFramesFITS(filenames, t - save_chunk, 1 + save_chunk * 2))
                np.save("./" + str(day_stamp) + "/timing-" + name_stamp + "-" + str(np.where(results == t)[0][0]) + ".npy", time_list[t - save_chunk:t + save_chunk])

    print datetime.datetime.now(), save_times
    count = len(np.where(results < 0)[0])
    print datetime.datetime.now(), "Rejected Stars: " + str(round(count * 100. / star_num, 2)) + "%"

    ind = np.where(results > 0)[0]

    if len(ind) > 0:  # if any events detected
        '''if os.path.exists('./fields/25mswcs.fits'):
            print datetime.datetime.now(), "WCS exists for field"
        else:
            print datetime.datetime.now(), "WCS not found, skipping..."
            return'''
        # the following was used in Emily's analysis, but will now be done externally after finding events
        #camCompare(ind, results, data, name_stamp, day_stamp, directory)
        pass
    else:
        print datetime.datetime.now(), "No events detected"
        # TODO: file deletion

    print datetime.datetime.now(), "Total stars in file:", star_num
    print datetime.datetime.now(), "Candidate events in file:", len(ind)
    print datetime.datetime.now(), "Closing:", file
    print "\n"

# fits folder to contain folders of fits files. what was a vid is now a folder
directory = './fits/'  # './emccd/'
folder_list = glob(directory + '*/')
field = '25ms'
folder_list.sort(key=getNum)
print folder_list
# remove non-identifying information in file name
#for x in range(0, len(folder_list)):
#    folder_list[x] = folder_list[x][len(directory)]#:-4]
for f in range(0, len(folder_list)-1):
    main(folder_list[f], field)  # run pipeline for each folder in the FITS directory
    gc.collect()

'''
""" process backlog """
# TODO: add third directory

# TODO: new directory names
directoryA = '/emccd1/'  # directory with EMCCD1 data
directoryB = '/emccd2/'  # directory with EMCCD2 data

# TODO: new file format
file_list_A = glob(directoryA + '*.vid')
file_list_B = glob(directoryB + '*.vid')

# order files from oldest to newest
file_list_A.sort(key=getNum)  
file_list_B.sort(key=getNum)

# remove non-identifying information in file name
for x in range(0, len(file_list_A)):
    file_list_A[x] = file_list_A[x][len(directoryA):-4]  # remove directory and file type from name
for x in range(0, len(file_list_B)):
    file_list_B[x] = file_list_B[x][len(directoryB):-4]

# TODO: run them separate or concurrently?
for f in range(0, len(file_list_A)):
    for half in range(0, 2):
        main(file_list_A[f], half, directoryA)  # run pipeline for each file in the EMCCD1 directory
        gc.collect()
        main(file_list_B[f], half, directoryB)  # run pipeline for each file in the EMCCD2 directory
        gc.collect()

""" once backlog data is processed, continuously check for new data """

while True:
    file_list_2A = glob(directoryA + '*vid')
    file_list_2B = glob(directoryB + '*.vid')
    # order files from oldest to newest
    file_list_2A.sort(key=getNum)
    file_list_2B.sort(key=getNum)

    # remove non-identifying information in file name
    for x in range(0, len(file_list_2A)):
        file_list_2A[x] = file_list_2A[x][len(directoryA):-4]
    for x in range(0, len(file_list_2B)):
        file_list_2B[x] = file_list_2B[x][len(directoryB):-4]

    # if new files
    if set(file_list_2A).issubset(set(file_list_A)) is False:
        diffA = list(set(file_list_2A).difference(set(file_list_A)))  # get list of new EMCCD1 data
        diffB = list(set(file_list_2B).difference(set(file_list_B)))  # get list of new EMCCD2 data
        # order files from oldest to newest
        diffA.sort(key=getNum)
        diffB.sort(key=getNum)
        # TODO: still need to handle uneven timing?
        upper = max((len(diffA), len(diffB)))
        for f in range(0, upper):
            for half in range(0, 2):
                try:
                    main(diffA[f], half, directoryA)  # process new EMCCD1 data
                except IndexError:  # handles uneven timing between the two cameras
                    pass
                gc.collect()
                try:
                    main(diffB[f], half, directoryB)  # process new EMCCD2 data
                except IndexError:
                    pass
                gc.collect()
        file_list_A = file_list_2A  # mark that all of file_list_2A has been processed
        file_list_B = file_list_2B
    else:
        print datetime.datetime.now(), "No new files"
        time.sleep(3600)  # if no new files, wait an hour before checking again
'''