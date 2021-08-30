from __future__ import absolute_import

from got10k.experiments import *

# from siamfc.siamfc import TrackerSiamFC
from siamrpn.tracker import SiamRPNTracker


if __name__ == '__main__':
    # setup tracker
    net_path = 'pretrained/siamrpn/siamrpn_50.pth'
    tracker = SiamRPNTracker(net_path)

    # setup experiments
    e = ExperimentGOT10k('data/GOT-10k', subset='val')
    

    # run tracking experiments and report performance
    e.run(tracker, visualize=False)
    prec_score, succ_score, succ_rate=e.report([tracker.name])

    ss = '-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(prec_score), float(succ_score), float(succ_rate))
    print(net_path.split('/')[-1], ss)
