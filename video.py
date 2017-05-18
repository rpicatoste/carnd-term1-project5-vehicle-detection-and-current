
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import os 
import datetime
import time

import search_and_classify as sc

directory = os.path.dirname("results/")
if not os.path.exists(directory):
    os.makedirs(directory)
        

# next video
video = 'project_video'
#video = 'test_video'
project_video_result_file = 'results\\'

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        
counter = 0


if('clip' in vars() or 'clip' in globals()):
    del clip
    
clip = VideoFileClip( video + '.mp4' )
project_video_result = clip.fl_image( sc.pipeline_heat ) 
 
project_video_result.write_videofile('results\\' + video + '_result_' + timestamp + '.mp4', audio=False)
print("Video " + video + " done")

