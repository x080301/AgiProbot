# Github for Project AgiProbot

Agiprobot is a project sponsored by the Carl Zeiss Foundation and implemented by KIT.
	
The project's primary objective is to develop an agile production system capable of dynamically adapting to uncertain product specifications through the application of artificial intelligence. To address this challenge, an interdisciplinary research group has been formed, comprising members from various institutes in mechanical engineering, electrical engineering, information technology, and computer science. This collaborative effort aims to harness complementary expertise effectively.
	
I joined this project as a research assistant and later and later as a student working on my graduation project. My role primarily focused on the autonomous disassembly of motors. I was tasked with designing an algorithm that, based on point cloud data acquired from other subsystems, could provide relevant parameters concerning the motor, especially the 6D pose of bolts. Additionally, I conducted theoretical research in related areas.![image](https://github.com/x080301/AgiProbot/assets/41547659/a55a3973-e168-4e54-bd0d-b314a3d9b8ad)


more details see READMEs in each folder.

# Structure

```
.
├─ agi_para_extracter                -- extracting screw position and further parameters of small motor
├─ binary_segmentation               -- pretraining and fine tuning for binary segmentation of zivid-scaned model, PCT
├─ large_motor_segmentation          -- pretraining and fine tuning for semantic segmentation of full model, PCT
|    ├─ blender                           --generating synthetic large motor data set with blender 
|    └─ ...
├─ .gitattributes
├─ .gitignore
└─ README.md
```
