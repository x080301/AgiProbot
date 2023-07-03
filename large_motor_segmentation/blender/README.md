# AgiProbot Synthetic Dataset Generator

This python library introduces a method, which uses blender to generate synthetic large motor data set for project
Agiprobot from existing blender models.

# Environments Requirement

Blender = 3.5.1

Python = 3.9

Scipy = 1.9.3

Open3d = 0.16

tpdm

# Method

This method is based on existing blender models, which are composed of several objects. Several objects form one
semantic segment.

First, all objects are joined together and then remeshed to generate a point cloud without labels (full model point
cloud)
so that the boundaries between objects will not appear in the result point cloud.

Then, objects in each semantic segment are output to each point cloud(segment point clouds). The color of the point
cloud is determined by the segment label.

Finally, The color of each point in the full model point cloud is set to the color of the closest point in segment point
clouds. The colorized full model point clouds form the required synthetic large motor dataset.

# Usage

Import function generate_pcd_with_label from this Library, run the function and get the results.

```
# Example:
from blender import generate_pcd_with_label

generate_pcd_with_label('E:/your_blender_model_dir', 'E:/result_dir')

```

All models in *.blend in 'E:/your_blender_model_dir' will be found out. And *.pcd files generated from them will be
saved to 'E:/result_dir'.

# Library Structure

```
│  __init__.py
│  README.md
│  generate_syn_dataset_with_blender.py             -- main pyfile
├─blender_scripts
│      generate_obj_with_label_using_blender.py     -- blender script, generate full model point cloud
│      pyscript_in_blender.py                       -- blender scriot, generate segment point clouds
```
