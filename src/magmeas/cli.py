"""Command line interface definition."""

import argparse
import glob
from pathlib import Path

from magmeas import VSM, __version__


def cli():
    """
    Command line interface.

    Calling "magmeas filename" at the command line, where "filename" is
    the full path to a DAT-file from a Quantum Design PPMS or Dynacools sytem,
    uses the VSM class to print the extrinsic magnetic properties and
    optionally dump them in a json file.
    """
    d_0 = (
        "Prints and/or stores various extrinsic magnetic properties"
        "calculated from a VSM measurement."
    )
    parser = argparse.ArgumentParser(description=d_0)

    path_help = (
        "full path(s) to any number of single .DAT file(s) or "
        "folder(s) containing several .DAT files. The files must "
        "have been produced by Quantum Design's PPMS or Dynacool system and "
        "contain specimen formatted as specified by magmeas."
        "In the case of folders, all the .tif images within each folder"
        "will be processed."
    )
    parser.add_argument("file_path", action="store", nargs="+", help=path_help)

    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    dump_help = (
        "dump selected magnetic properties to json. One json file per DAT-file"
        " will be stored in the same directory as the DAT-file. "
        "The name of each json file is the corresponding DAT-file name "
        "(without the .DAT extension) plus _properties.json"
    )
    parser.add_argument(
        "-d",
        "--dump",
        action="store_true",
        help=dump_help,
    )

    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="do not print output to terminal",
    )

    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="plot data, save as figure if combined with --dump",
    )

    parser.add_argument(
        "--hdf5",
        action="store_true",
        help="save DAT-file contents and calculated properties to hdf5",
    )

    args = parser.parse_args()

    if args.silent and not args.dump:
        err_msg = (
            "Printing and saving both suppressed (-s but no -d): " "-> nothing to do."
        )
        parser.error(err_msg)

    for p_dat in args.file_path:
        p_dat = Path(p_dat)
        if p_dat.is_file():
            vsm_dat = VSM(p_dat)
            if not args.silent:
                vsm_dat.print_properties()
            if args.dump:
                fn = p_dat.parent.joinpath(p_dat.stem + "_properties.json")
                vsm_dat.properties_to_json(fn)
            if args.plot and args.dump:
                fn = p_dat.parent.joinpath(p_dat.stem + "_plot.png")
                vsm_dat.plot(filepath=fn)
            if args.plot and not args.dump:
                vsm_dat.plot()
            if args.hdf5:
                vsm_dat.to_hdf5()
        else:
            all_dats = glob.glob(p_dat.joinpath("*.DAT").as_posix())
            for i in all_dats:
                i = Path(i)
                vsm_dat = VSM(i)
                if not args.silent:
                    vsm_dat.print_properties()
                if args.dump:
                    fn = i.parent.joinpath(i.stem + "_properties.json")
                    vsm_dat.properties_to_json(fn)
                if args.plot and args.dump:
                    fn = i.parent.joinpath(i.stem + "_plot.png")
                    vsm_dat.plot(filepath=fn)
                if args.plot and not args.dump:
                    vsm_dat.plot()
                if args.hdf5:
                    vsm_dat.to_hdf5()
