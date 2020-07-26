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
    def __init__(self, Data_4D, Data_ADF, calib_pm, voltage, aperture):
        self.data_adf = Data_ADF
        self.data_4D = Data_4D
        self.calib = calib_pm
        self.voltage = voltage
        self.wavelength = st.sim.wavelength_ang(voltage) * 100
        self.aperture = aperture

    def show_BF_ADF(self, imsize=(20, 10)):
        fontsize = int(np.amax(np.asarray(imsize)))
        plt.figure(figsize=imsize)
        plt.subplot(1, 2, 1)
        plt.imshow(self.data_adf)
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
        plt.imshow(np.sum(self.data_4D, axis=(-1, -2)))
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

    def get_cbed(self, imsize=(15, 15)):
        self.cbed = np.median(self.data_4D, axis=(0, 1))
        self.beam_x, self.beam_y, self.beam_r = st.util.sobel_circle(self.cbed)
        self.inverse = self.aperture / (self.beam_r * self.wavelength)
        plt.figure(figsize=imsize)
        plt.imshow(self.cbed)
        scalebar = mpss.ScaleBar(self.inverse, "1/pm", mpss.SI_LENGTH_RECIPROCAL)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        plt.gca().add_artist(scalebar)
        plt.axis("off")

    def initial_dpc(self, imsize=(30, 15)):
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

        vm = np.amax(np.abs(np.concatenate((self.XCom, self.YCom), axis=1)))
        fontsize = int(np.amax(np.asarray(imsize)))
        sc_font = {"weight": "bold", "size": fontsize}

        fig = plt.figure(figsize=imsize)
        gs = mpgs.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        im = ax1.imshow(self.XCom, vmin=-vm, vmax=vm, cmap="RdBu_r")
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

        im = ax2.imshow(self.YCom, vmin=-vm, vmax=vm, cmap="RdBu_r")
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

        p1 = ax1.get_position().get_points().flatten()
        p2 = ax2.get_position().get_points().flatten()

        ax_cbar = fig.add_axes([p1[0] - 0.075, -0.01, p2[2], 0.02])
        cbar = plt.colorbar(im, cax=ax_cbar, orientation="horizontal")
        cbar.set_label(r"$\mathrm{Beam\: Shift\: \left(pm^{-1}\right)}$", **sc_font)

    def correct_dpc(self, imsize=(30, 15)):
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

        vm = np.amax(np.abs(np.concatenate((self.XComC, self.YComC), axis=1)))
        fontsize = int(np.amax(np.asarray(imsize)))
        sc_font = {"weight": "bold", "size": fontsize}

        fig = plt.figure(figsize=imsize)
        gs = mpgs.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        im = ax1.imshow(self.XComC, vmin=-vm, vmax=vm, cmap="RdBu_r")
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

        im = ax2.imshow(self.YComC, vmin=-vm, vmax=vm, cmap="RdBu_r")
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

        p1 = ax1.get_position().get_points().flatten()
        p2 = ax2.get_position().get_points().flatten()

        ax_cbar = fig.add_axes([p1[0] - 0.075, -0.01, p2[2], 0.02])
        cbar = plt.colorbar(im, cax=ax_cbar, orientation="horizontal")
        cbar.set_label(r"$\mathrm{Beam\: Shift\: \left(pm^{-1}\right)}$", **sc_font)

    def show_charge(self, imsize=(15, 15)):
        fontsize = int(np.amax(np.asarray(imsize)))
        XComV = self.XComC * self.wavelength * self.voltage
        YComV = self.YComC * self.wavelength * self.voltage
        self.charge = (-1) * (
            (np.gradient(XComV)[1] + np.gradient(YComV)[0])
            / (self.calib * (10 ** (-12)))
        )
        cm = np.amax(np.abs(self.charge))
        plt.figure(figsize=imsize)
        plt.imshow(self.charge, vmin=-cm, vmax=cm, cmap="seismic")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        at = mploff.AnchoredText(
            "Charge from DPC", prop=dict(size=fontsize), frameon=True, loc="lower left"
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        plt.gca().add_artist(at)
        plt.tight_layout()

    def show_potential(self, imsize=(15, 15)):
        fontsize = int(np.amax(np.asarray(imsize)))
        XComV = self.XComC * self.wavelength * self.voltage
        YComV = self.YComC * self.wavelength * self.voltage
        self.pot = st.dpc.integrate_dpc(XComV, YComV)
        cm = np.amax(np.abs(self.pot))
        plt.figure(figsize=imsize)
        plt.imshow(self.pot, vmin=-cm, vmax=cm, cmap="BrBG_r")
        scalebar = mpss.ScaleBar(self.calib / 1000, "nm")
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        at = mploff.AnchoredText(
            "Measured potential",
            prop=dict(size=fontsize),
            frameon=True,
            loc="lower left",
        )
        at.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        plt.gca().add_artist(at)
        plt.tight_layout()

    def plot_color_dpc(self, skip=2, portion=7, imsize=(20, 10)):
        fontsize = int(np.amax(np.asarray(imsize)))
        sc_font = {"weight": "bold", "size": fontsize}
        mpl.rc("font", **sc_font)
        cc = self.XComC + ((1j) * self.YComC)
        cc_color = st.util.cp_image_val(cc)
        cutter = 1 / portion
        cutstart = (
            np.round(
                np.asarray(self.XComC.shape) - (cutter * np.asarray(self.XComC.shape))
            )
        ).astype(int)
        ypos, xpos = np.mgrid[0 : self.YComC.shape[0], 0 : self.XComC.shape[1]]
        ypos = ypos
        xcut = (
            xpos[cutstart[0] : self.XComC.shape[0], cutstart[1] : self.XComC.shape[1]]
            - cutstart[1]
        )
        ycut = (
            np.flipud(
                ypos[
                    cutstart[0] : self.XComC.shape[0], cutstart[1] : self.XComC.shape[1]
                ]
            )
            - cutstart[0]
        )
        dx = self.XComC[
            cutstart[0] : self.XComC.shape[0], cutstart[1] : self.XComC.shape[1]
        ]
        dy = self.YComC[
            cutstart[0] : self.XComC.shape[0], cutstart[1] : self.XComC.shape[1]
        ]
        cc_cut = cc_color[
            cutstart[0] : self.XComC.shape[0], cutstart[1] : self.XComC.shape[1], :
        ]

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

        plt.subplot(1, 2, 2)
        plt.imshow(cc_cut)
        plt.quiver(
            xcut[::skip, ::skip],
            ycut[::skip, ::skip],
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
