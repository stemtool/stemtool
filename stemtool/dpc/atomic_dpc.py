import scipy.ndimage as scnd
import scipy.optimize as sio
import numpy as np
import stemtool as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib_scalebar.scalebar as mpss
import matplotlib.offsetbox as mploff
import matplotlib.gridspec as mpgs
import matplotlib as mpl


class atomic_dpc(object):
    """
    Atomic Resolution DPC estimation

    Parameters
    ----------
    Data_4D:  ndarray
              Four-dimensional dataset where the first two
              dimensions are real space scanning dimensions,
              while the last two dimenions are the Fourier
              space electron diffraction patterns
    Data_ADF: ndarray
              Simultaneously collected two-dimensional ADF-STEM
              image
    calib_pm: float
              Real space pixel calibration in picometers
    voltage:  float
              Microscope accelerating voltage in kV
    aperture: float
              The probe forming condenser aperture in milliradians

    Notes
    -----
    This class function takes in a 4D-STEM image, and a simultaneously
    collected atomic resolution ADF-STEM image. Based on the accelerating
    voltage and the condenser aperture this calculates the center of mass
    (C.O.M.) shifts in the central undiffracted beam. Using the idea that
    the curl of the beam shift vectors, should be minimized at the correct
    Fourier rotation angles, this class also corrects for rotation of the
    collceted 4D-STEM data with respect to the optic axis. Using these, a
    correct potential accumulation and charge accumulation maps could be
    built. To prevent errors, we convert everything to SI units first.

    Examples
    --------
    Run as:

    >>> DPC = st.dpc.atomic_dpc(Data_4D, DataADF, calibration, voltage, aper)

    Once the data is loaded, the ADF-STEM and the BF-STEM images could be
    visualized as:

    >>> DPC.show_ADF_BF()

    Then the following call generates the mean CBED image, and if the show_image
    call is True, shows the mean image.

    >>> DPC.get_cbed(show_image = True)

    The initial uncorrected DPC shifts are generated as:

    >>> DPC.initial_dpc()

    The corrected DPC shifts are generated:

    >>> DPC.correct_dpc()

    The charge map is generated through:

    >>> DPC.show_charge()

    While the potential map is generated though:

    >>> DPC.show_potential()

    If a section of the image needs to be observed, to visualize the beam shifts,
    call the following:

    >>> DPC.plot_color_dpc()

    References
    ----------
    .. [1] Müller, K. et al. "Atomic electric fields revealed by a quantum mechanical
        approach to electron picodiffraction". Nat. Commun. 5:565303 doi: 10.1038/ncomms6653 (2014)
    .. [2] Savitzky, Benjamin H., Lauren A. Hughes, Steven E. Zeltmann, Hamish G. Brown,
        Shiteng Zhao, Philipp M. Pelz, Edward S. Barnard et al. "py4DSTEM: a software package for
        multimodal analysis of four-dimensional scanning transmission electron microscopy datasets."
        arXiv preprint arXiv:2003.09523 (2020).
    .. [3] Ishizuka, Akimitsu, Masaaki Oka, Takehito Seki, Naoya Shibata,
        and Kazuo Ishizuka. "Boundary-artifact-free determination of
        potential distribution from differential phase contrast signals."
        Microscopy 66, no. 6 (2017): 397-405.
    """

    def __init__(self, Data_4D, Data_ADF, calib_pm, voltage, aperture):
        """
        Load the user defined values.
        It also calculates the wavelength based on the accelerating voltage
        This also loads several SI constants as the following attributes

        `planck`:   The Planck's constant

        `epsilon0`: The dielectric permittivity of free space

        `e_charge`: The charge of an electron in Coulombs
        """
        self.data_adf = Data_ADF
        self.data_4D = Data_4D
        self.calib = calib_pm
        self.voltage = voltage * 1000  # convert to volts
        self.wavelength = st.sim.wavelength_ang(voltage) * (
            10 ** (-10)
        )  # convert to meters
        self.aperture = aperture / 1000  # convert to radians
        self.planck = 6.62607004 * (10 ** (-34))
        self.epsilon0 = 8.85418782 * (10 ** (-12))
        self.e_charge = (-1) * 1.60217662 * (10 ** (-19))
        e_mass = 9.109383 * (10 ** (-31))
        c = 299792458
        self.sigma = (
            (2 * np.pi / (self.wavelength * self.voltage))
            * ((e_mass * (c ** 2)) + (self.e_charge * self.voltage))
        ) / ((2 * e_mass * (c ** 2)) + (self.e_charge * self.voltage))

    def show_ADF_BF(self, imsize=(20, 10)):
        """
        The ADF-STEM image is already loaded, while the `data_bf`
        attribute is obtained by summing up the 4D-STEM dataset along it's
        Fourier dimensions. This is also a great checkpoint to see whether
        the ADF-STEM and the BF-STEM images are the inverse of each other.
        """
        self.data_bf = np.sum(self.data_4D, axis=(-1, -2))
        fontsize = int(np.amax(np.asarray(imsize)))
        plt.figure(figsize=imsize)
        plt.subplot(1, 2, 1)
        plt.imshow(self.data_adf, cmap="inferno")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 0
        scalebar.color = "w"
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        at = mploff.AnchoredText(
            "ADF-STEM", prop=dict(size=fontsize), frameon=True, loc="lower left"
        )
        at.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        plt.gca().add_artist(at)

        plt.subplot(1, 2, 2)
        plt.imshow(self.data_bf, cmap="inferno")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 0
        scalebar.color = "w"
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        at = mploff.AnchoredText(
            "Summed 4D-STEM", prop=dict(size=fontsize), frameon=True, loc="lower left"
        )
        at.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        plt.gca().add_artist(at)
        plt.tight_layout()

    def get_cbed(self, imsize=(15, 15), show_image=False):
        """
        We calculate the mean CBED pattern by averaging the Fourier data, to
        get the object attribute `cbed`. We fit this with a circle function to
        obtain the object attributes:

        `beam_x`: x-coordinates of the circle

        `beam_y`: y-coordinates of the circle

        `beam_r`: radius of the circle

        We use the calculated radius and the known aperture size to get the Fourier
        space calibration, which is stored as the `inverse` attribute
        """
        self.cbed = np.mean(self.data_4D, axis=(0, 1))
        self.beam_x, self.beam_y, self.beam_r = st.util.sobel_circle(self.cbed)
        self.inverse = self.aperture / (self.beam_r * self.wavelength)
        if show_image:
            plt.figure(figsize=imsize)
            plt.imshow(self.cbed, cmap="inferno")
            scalebar = mpss.ScaleBar(self.inverse, "1/m", mpss.SI_LENGTH_RECIPROCAL)
            scalebar.location = "lower right"
            scalebar.box_alpha = 1
            scalebar.color = "k"
            plt.gca().add_artist(scalebar)
            plt.axis("off")

    def initial_dpc(self, imsize=(30, 17), normalize=True):
        """
        This calculates the initial DPC center of mass shifts by measuring
        the center of mass of each image in the 4D-STEM dataset, and then
        comparing that center of mass with the average disk center of the
        entire dataset.
        """
        qq, pp = np.mgrid[0 : self.data_4D.shape[-1], 0 : self.data_4D.shape[-2]]
        yy, xx = np.mgrid[0 : self.data_4D.shape[0], 0 : self.data_4D.shape[1]]
        yy = np.ravel(yy)
        xx = np.ravel(xx)
        self.YCom = np.empty(self.data_4D.shape[0:2], dtype=np.float)
        self.XCom = np.empty(self.data_4D.shape[0:2], dtype=np.float)
        for ii in range(len(yy)):
            pattern = self.data_4D[yy[ii], xx[ii], :, :]
            self.YCom[yy[ii], xx[ii]] = self.inverse * (
                (np.sum(np.multiply(qq, pattern)) / np.sum(pattern)) - self.beam_y
            )
            self.XCom[yy[ii], xx[ii]] = self.inverse * (
                (np.sum(np.multiply(pp, pattern)) / np.sum(pattern)) - self.beam_x
            )
        if normalize:
            self.YCom = self.YCom - np.mean(self.YCom)
            self.XCom = self.XCom - np.mean(self.XCom)

        vm = (np.amax(np.abs(np.concatenate((self.XCom, self.YCom), axis=1)))) / (
            10 ** 9
        )
        fontsize = int(0.9 * np.amax(np.asarray(imsize)))
        sc_font = {"weight": "bold", "size": fontsize}

        plt.figure(figsize=imsize)
        gs = mpgs.GridSpec(imsize[1], imsize[0])
        ax1 = plt.subplot(gs[0:15, 0:15])
        ax2 = plt.subplot(gs[0:15, 15:30])
        ax3 = plt.subplot(gs[15:17, :])

        ax1.imshow(self.XCom / (10 ** 9), vmin=-vm, vmax=vm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax1.add_artist(scalebar)
        at = mploff.AnchoredText(
            "Shift in X direction",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax1.add_artist(at)
        ax1.axis("off")

        ax2.imshow(self.YCom / (10 ** 9), vmin=-vm, vmax=vm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax2.add_artist(scalebar)
        at = mploff.AnchoredText(
            "Shift in Y direction",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax2.add_artist(at)
        ax2.axis("off")

        sb = np.zeros((10, 1000), dtype=np.float)
        for ii in range(10):
            sb[ii, :] = np.linspace(-vm, vm, 1000)
        ax3.imshow(sb, cmap="RdBu_r")
        ax3.yaxis.set_visible(False)
        x1 = np.linspace(0, 1000, 8)
        ax3.set_xticks(x1)
        ax3.set_xticklabels(np.round(np.linspace(-vm, vm, 8), 2))
        for axis in ["top", "bottom", "left", "right"]:
            ax3.spines[axis].set_linewidth(2)
            ax3.spines[axis].set_color("black")
        ax3.xaxis.set_tick_params(width=2, length=6, direction="out", pad=10)
        ax3.set_title(r"$\mathrm{Beam\: Shift\: \left(nm^{-1}\right)}$", **sc_font)
        plt.tight_layout()

    def correct_dpc(self, imsize=(30, 17)):
        """
        This corrects for the rotation angle of the pixellated detector
        with respect to the optic axis. Some pixellated detectors flip
        the image, and if there is an image flip, it corrects it too.
        The mechanism of this, we compare the gradient of both the flipped
        and the unflipped DPC data at multiple rotation angles, and the value
        that has the highest relative contrast with the ADF-STEM image is taken
        as 90 degrees from the correct angle.
        """
        flips = np.zeros(4, dtype=bool)
        flips[2:4] = True
        chg_sums = np.zeros(4, dtype=self.XCom.dtype)
        angles = np.zeros(4, dtype=self.YCom.dtype)
        x0 = 90
        for ii in range(2):
            to_flip = flips[2 * ii]
            if to_flip:
                xdpcf = np.flip(self.XCom)
            else:
                xdpcf = self.XCom
            rho_dpc, phi_dpc = st.dpc.cart2pol(self.XCom, self.YCom)
            x = sio.minimize(st.dpc.angle_fun, x0, args=(rho_dpc, phi_dpc))
            min_x = x.x
            sol1 = min_x - 90
            sol2 = min_x + 90
            chg_sums[int(2 * ii)] = np.sum(
                st.dpc.charge_dpc(xdpcf, self.YCom, sol1) * self.data_adf
            )
            chg_sums[int(2 * ii + 1)] = np.sum(
                st.dpc.charge_dpc(xdpcf, self.YCom, sol2) * self.data_adf
            )
            angles[int(2 * ii)] = sol1
            angles[int(2 * ii + 1)] = sol2
        self.angle = (-1) * angles[chg_sums == np.amin(chg_sums)][0]
        self.final_flip = flips[chg_sums == np.amin(chg_sums)][0]

        if self.final_flip:
            xdpcf = np.fliplr(self.XCom)
        else:
            xdpcf = np.copy(self.XCom)
        rho_dpc, phi_dpc = st.dpc.cart2pol(xdpcf, self.YCom)
        self.XComC, self.YComC = st.dpc.pol2cart(
            rho_dpc, (phi_dpc - (self.angle * ((np.pi) / 180)))
        )

        vm = (np.amax(np.abs(np.concatenate((self.XComC, self.YComC), axis=1)))) / (
            10 ** 9
        )
        fontsize = int(0.9 * np.max(imsize))
        sc_font = {"weight": "bold", "size": fontsize}

        plt.figure(figsize=imsize)

        gs = mpgs.GridSpec(imsize[1], imsize[0])
        ax1 = plt.subplot(gs[0:15, 0:15])
        ax2 = plt.subplot(gs[0:15, 15:30])
        ax3 = plt.subplot(gs[15:17, :])

        ax1.imshow(self.XComC / (10 ** 9), vmin=-vm, vmax=vm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax1.add_artist(scalebar)
        at = mploff.AnchoredText(
            "Corrected shift in X direction",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax1.add_artist(at)
        ax1.axis("off")

        ax2.imshow(self.YComC / (10 ** 9), vmin=-vm, vmax=vm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax2.add_artist(scalebar)
        at = mploff.AnchoredText(
            "Corrected shift in Y direction",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax2.add_artist(at)
        ax2.axis("off")

        sb = np.zeros((10, 1000), dtype=np.float)
        for ii in range(10):
            sb[ii, :] = np.linspace(-vm, vm, 1000)
        ax3.imshow(sb, cmap="RdBu_r")
        ax3.yaxis.set_visible(False)
        x1 = np.linspace(0, 1000, 8)
        ax3.set_xticks(x1)
        ax3.set_xticklabels(np.round(np.linspace(-vm, vm, 8), 2))
        for axis in ["top", "bottom", "left", "right"]:
            ax3.spines[axis].set_linewidth(2)
            ax3.spines[axis].set_color("black")
        ax3.xaxis.set_tick_params(width=2, length=6, direction="out", pad=10)
        ax3.set_title(r"$\mathrm{Beam\: Shift\: \left(nm^{-1}\right)}$", **sc_font)
        plt.tight_layout()

        self.MomentumX = self.planck * self.XComC
        self.MomentumY = self.planck * self.YComC
        # assuming infinitely thin sample
        self.e_fieldX = self.MomentumX / self.e_charge
        self.e_fieldY = self.MomentumY / self.e_charge

    def show_charge(self, imsize=(15, 17)):
        """
        We calculate the charge from the corrected DPC
        center of mass datasets. This is done through
        Poisson's equation.
        """
        fontsize = int(np.amax(np.asarray(imsize)))

        # Use Poisson's equation
        self.charge = (
            (
                (np.gradient(self.e_fieldX)[1] + np.gradient(self.e_fieldY)[0])
                * (self.calib * (10 ** (-12)))
            )
            * self.epsilon0
            * 4
            * np.pi
        )
        cm = np.amax(np.abs(self.charge))
        plt.figure(figsize=imsize)
        fontsize = int(0.9 * np.max(imsize))
        sc_font = {"weight": "bold", "size": fontsize}

        gs = mpgs.GridSpec(imsize[1], imsize[0])
        ax1 = plt.subplot(gs[0:15, 0:15])
        ax2 = plt.subplot(gs[15:17, :])

        ax1.imshow(self.charge, vmin=-cm, vmax=cm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax1.add_artist(scalebar)
        ax1.axis("off")
        at = mploff.AnchoredText(
            "Charge from DPC", prop=dict(size=fontsize), frameon=True, loc="lower left"
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at)

        sb = np.zeros((10, 1000), dtype=np.float)
        for ii in range(10):
            sb[ii, :] = np.linspace(cm / self.e_charge, -(cm / self.e_charge), 1000)
        ax2.imshow(sb, cmap="RdBu_r")
        ax2.yaxis.set_visible(False)
        no_labels = 7
        x1 = np.linspace(0, 1000, no_labels)
        ax2.set_xticks(x1)
        ax2.set_xticklabels(
            np.round(
                np.linspace(cm / self.e_charge, -(cm / self.e_charge), no_labels), 6
            )
        )
        for axis in ["top", "bottom", "left", "right"]:
            ax2.spines[axis].set_linewidth(2)
            ax2.spines[axis].set_color("black")
        ax2.xaxis.set_tick_params(width=2, length=6, direction="out", pad=10)
        ax2.set_title(r"$\mathrm{Charge\: Density\: \left(e^{-} \right)}$", **sc_font)

        plt.tight_layout()

    def show_potential(self, imsize=(15, 17)):
        """
        Calculate the projected potential from the DPC measurements.
        This is accomplished by calculating the phase shift iteratively
        from the normalized center of mass shifts. Normalization means
        calculating COM shifts in inverse length units and then multiplying
        them with the electron wavelength to get an electron independent
        mrad shift, which is used to generate the phase. This phase is
        proportional to the projected potential for weak phase object
        materials (with *lots* of assumptions)
        """
        fontsize = int(np.amax(np.asarray(imsize)))
        self.phase = st.dpc.integrate_dpc(
            self.XComC * self.wavelength, self.YComC * self.wavelength
        )
        self.potential = self.phase / self.sigma

        pm = np.amax(np.abs(self.potential)) * (10 ** 10)
        plt.figure(figsize=imsize)
        fontsize = int(0.9 * np.max(imsize))
        sc_font = {"weight": "bold", "size": fontsize}

        gs = mpgs.GridSpec(imsize[1], imsize[0])
        ax1 = plt.subplot(gs[0:15, 0:15])
        ax2 = plt.subplot(gs[15:17, :])

        ax1.imshow(self.potential * (10 ** 10), vmin=-pm, vmax=pm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax1.add_artist(scalebar)
        ax1.axis("off")
        at = mploff.AnchoredText(
            "Calculated projected potential from DPC phase",
            prop=dict(size=fontsize),
            frameon=True,
            loc="lower left",
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at)

        sb = np.zeros((10, 1000), dtype=np.float)
        for ii in range(10):
            sb[ii, :] = np.linspace(-pm, pm, 1000)
        ax2.imshow(sb, cmap="RdBu_r")
        ax2.yaxis.set_visible(False)
        no_labels = 7
        x1 = np.linspace(0, 1000, no_labels)
        ax2.set_xticks(x1)
        ax2.set_xticklabels(np.round(np.linspace(-pm, pm, no_labels), 6))
        for axis in ["top", "bottom", "left", "right"]:
            ax2.spines[axis].set_linewidth(2)
            ax2.spines[axis].set_color("black")
        ax2.xaxis.set_tick_params(width=2, length=6, direction="out", pad=10)
        ax2.set_title(r"Projected Potential (VÅ)", **sc_font)

        plt.tight_layout()

    def plot_color_dpc(self, start_frac=0, size_frac=1, skip=2, imsize=(20, 10)):
        """
        Use this to plot the corrected DPC center of mass shifts. If no variables
        are passed, the arrows are overlaid on the entire image.

        Parameters
        ----------
        start_frac: float, optional
                    The starting fraction of the image, where you will cut from
                    to show the overlaid arrows. Default is 0
        stop_frac:  float, optional
                    The ending fraction of the image, where you will cut from
                    to show the overlaid arrows. Default is 1
        """
        fontsize = int(np.amax(np.asarray(imsize)))
        sc_font = {"weight": "bold", "size": fontsize}
        mpl.rc("font", **sc_font)
        cc = self.XComC + ((1j) * self.YComC)
        cc_color = st.util.cp_image_val(cc)
        cutstart = (np.asarray(self.XComC.shape) * start_frac).astype(int)
        cut_stop = (np.asarray(self.XComC.shape) * (start_frac + size_frac)).astype(int)
        ypos, xpos = np.mgrid[0 : self.YComC.shape[0], 0 : self.XComC.shape[1]]
        ypos = ypos
        xcut = xpos[cutstart[0] : cut_stop[0], cutstart[1] : cut_stop[1]]
        ycut = np.flipud(ypos[cutstart[0] : cut_stop[0], cutstart[1] : cut_stop[1]])
        dx = self.XComC[cutstart[0] : cut_stop[0], cutstart[1] : cut_stop[1]]
        dy = self.YComC[cutstart[0] : cut_stop[0], cutstart[1] : cut_stop[1]]
        cc_cut = cc_color[cutstart[0] : cut_stop[0], cutstart[1] : cut_stop[1]]

        overlay = mpl.patches.Rectangle(
            cutstart[0:2],
            cut_stop[0] - cutstart[0],
            cut_stop[1] - cutstart[1],
            linewidth=1.5,
            edgecolor="w",
            facecolor="none",
        )

        plt.figure(figsize=imsize)
        plt.subplot(1, 2, 1)
        plt.imshow(cc_color)
        scalebar = mpss.ScaleBar(self.calib, "pm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 0
        scalebar.color = "w"
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        at = mploff.AnchoredText(
            "Center of Mass Shift",
            prop=dict(size=fontsize),
            frameon=True,
            loc="lower left",
        )
        at.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        plt.gca().add_artist(at)
        plt.gca().add_patch(overlay)

        plt.subplot(1, 2, 2)
        plt.imshow(cc_cut)
        plt.quiver(
            xcut[::skip, ::skip] - cutstart[1],
            ycut[::skip, ::skip] - cutstart[0],
            dx[::skip, ::skip],
            dy[::skip, ::skip],
            pivot="mid",
            color="w",
        )
        scalebar = mpss.ScaleBar(self.calib, "pm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 0
        scalebar.color = "w"
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        plt.tight_layout()
