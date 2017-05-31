import sys, os
import numpy as np
from elementtree.SimpleXMLWriter import XMLWriter
from eagleeye import Memset, EasyArgs, EasyConfig, Theta


def usage():
    print "usage: mapping.py -trainer <trainer xml> -output <output dataset> [<multiple csv files>] {-map_trainer_mode | -force_side <buttonside|backside> | -config <file>}"


def main(sysargs):
    args = EasyArgs(sysargs)
    cfg = EasyConfig(args.config, group="mapper")

    if "help" in args:
        usage()
        return 0

    if ["trainer", "output"] not in args:
        print "Must specify: -trainer, -output files"
        usage()
        return 1

    if len(args) == 1:
        print "Not enough input CSV files"
        usage()
        return 1

    if len(args) > 2 and args.map_trainer_mode:
        print "Too many CSV for trainer-mapping mode"
        usage()
        return 1

    if "force_side" in args:
        side = Theta.resolve(args.force_side)
        if side == Theta.NonDual:
            print "Invalid force_side argument:", args.force_side
            usage()
            return 1

        # set side overrides
        force_button = (side == Theta.Buttonside)
        force_back = not force_button
    else:
        force_button = force_back = False

    # working vars
    csvs = {}
    frame_num = 0

    # open source CSV datasets
    for i in range(1, len(args)):
        print args[i]
        csvs[i] = Memset(args[i])

    # reel all the files up to their first flash
    for i in csvs:
        csvs[i].restrict()
        if len(csvs[i].row()) < 10:
            print "CSV file:", args[i], "contains no marker data!\nAborting."
            return 1

    # override csv name
    if args.map_trainer_mode:
        csvs[1]._name = cfg.trainer_target

    # open calib files
    try:
        backside = (cfg, Theta.Backside)
    except Exception as e:
        print e.message
        return 1

    count = {'bts': 0, 'bks': 0, 'rej': 0}

    # open destination XML
    with open(args.output, "w") as xmlfile:
        w = XMLWriter(xmlfile)
        w.declaration()
        xmlfile.write("<!DOCTYPE dataset SYSTEM \"http://storage.gwillz.com.au/eagleeye_v2.dtd\">")
        doc = w.start("dataset")

        # iterates until all vicon csvs reach eof
        while True:
            w.start("frameInformation")
            w.element("frame", number=str(frame_num))

            # iterates through each vicon csv at the current row
            for i in csvs:
                side = 'this will default to buttonside when it really shouldn\'t...'
                c = csvs[i]
                # determine marker quality
                try:
                    max_reflectors = int(c.row()[8])
                    visible_reflectors = int(c.row()[9])
                except:
                    print "Error in reading quality at row {}".format(i)
                    return 1

                try:
                    # read VICON data
                    x = float(c.row()[2])
                    y = float(c.row()[3])
                    z = float(c.row()[4])

                    # TODO: is this necessary? We never use the object's rotation
                    rx = float(c.row()[5])
                    ry = float(c.row()[6])
                    rz = float(c.row()[7])
                except:
                    print "Error occurred when converting VICON data at row {}".format(i)
                    return 1

                # run projection/mapping on VICON data
                if backside.isVisible((x, y, z)):
                    points = backside.reprojpts((x, y, z))
                    side = 'backside'
                    count['bks'] += 1

                elif buttonside.isVisible((x, y, z)):
                    points = buttonside.reprojpts((x, y, z))
                    points[0] += 960  # add 960 to x for rightside points (Ricoh video is both frames side by side)
                    side = 'buttonside'
                    count['bts'] += 1

                # TODO don't write non visible dots?
                else:
                    points = [0., 0.]
                    count['rej'] += 1

                # TODO: Change DTD and double check with Manjung
                w.start("object", id=str(i), name=c.name(), lens=Theta.name(side))
                w.element("boxinfo", height="99", width="99", x=str(points[0] - 50), y=str(points[1] - 50))
                w.element("centroid", x=str(points[0]), y=str(points[1]), rx=str(rx), ry=str(ry), rz=str(rz))
                w.element("visibility", visible=str(visible_reflectors), visibleMax=str(max_reflectors))
                w.end()

            w.end()

            # test end of files
            eofs = 0
            for i in csvs:
                if csvs[i].eof(): eofs += 1

            if len(csvs) == eofs:
                print "end of all datasets"
                break

            # load next vicon frame
            frame_num += 1
            for i in csvs:
                csvs[i].next()

        w.close(doc)

        print "\nbuttonside", count['bts']
        print "backside", count['bks']
        print "rejected", count['rej']

    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
