import glob
import torch
from siamrpn import TrackerSiamRPN
import re
from util import *

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='video_frames/', help='datasets')
parser.add_argument('--net_path', type = str, default='model.pth', help='network path')
parser.add_argument('--object', nargs='*',action="store",help='set object region') ###### x,y,w,h x,y,w,h   || center, w,h
parser.add_argument('--region', nargs='*',action="store",help='set alarm region')
parser.add_argument('-v','--video',type = str, default = 'ARENA-W1-11_03_ENV_RGB_3.mp4', help='video name')
parser.add_argument('--type', type = int, default = 0, help = 'tracker type')
parser.add_argument('--boxnum', type = int, default = 1, help = 'box number')

args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #torch.backends.cudnn.benchmark = True

    # Setup Model

    ### init model ###
    tracker = TrackerSiamRPN(net_path=args.net_path)

    # Parse Image file
   # cap = cv2.VideoCapture(args.video)
    #if (cap.isOpened() == False):
   #     print('error opening video file')
   # ret, frame = cap.read()

    img_files = natural_sort(glob.glob(args.base_path + '*.jpg'))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    cv2.namedWindow("SiamRPN_Tracking", cv2.WND_PROP_AUTOSIZE)

    ### init the region
    region, object = SelectRoi(args.region, args.object, ims[0], number = args.boxnum)

    toc = 0
    f = 0
    #while(cap.isOpened()):
        #ret, frame = cap.read()
        #if ret == True:
        #else:
    for f, im in enumerate(ims):
        for idx in range(args.boxnum):
            cv2.polylines(im, [region[idx]], True, (0,0,255), 1)
        tic = cv2.getTickCount()
        if f == 0:  # init
            #target_pos = np.array([x + w / 2, y + h / 2])
            #target_sz = np.array([w, h])
            tracker.init(im, object)
            #state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'])  # init tracker
        elif f > 0:  # tracking
            box = tracker.update(im) ### x - w / 2,y - h /2,w,h ### (number, 4)
            for idx in range(box.shape[0]):
                x,y,w,h = box[idx].tolist()
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(im, [pts], True, (0, 255, 0), 3)
            #state = siamese_track(state, frame, mask_enable=True, refine_enable=True)  # track

            #location = state['ploygon'].flatten()
            #mask = state['mask'] > state['p'].seg_thr
            #im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            # pts = np.array([[x, y],[x + w, y], [x + w, y + h], [x, y + h]],np.int32)
            # pts = pts.reshape((-1,1,2))
            # cv2.polylines(im, [pts], True, (0, 255, 0), 3)
            cv2.imshow('SiamRPN_Tracking', im)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
#    cap.release()
    cv2.destroyAllWindows()
