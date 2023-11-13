# Project AgiProbot

Agiprobot is a project sponsored by the Carl Zeiss Foundation and implemented by KIT. 	
The project's primary objective is to develop an agile production system capable of dynamically adapting to uncertain product specifications through the application of artificial intelligence. To address this challenge, an interdisciplinary research group has been formed, comprising members from various institutes in mechanical engineering, electrical engineering, information technology, and computer science. This collaborative effort aims to harness complementary expertise effectively.

I joined this project as a research assistant and later as a student working on my master thesis. My role primarily focused on the autonomous disassembly of motors. I was tasked with designing an deep learning algorithm that, based on point cloud data acquired from other subsystems, could provide relevant parameters concerning the motor, especially the 6D pose of bolts. 

Additionally, I conducted theoretical research in related areas, such as point cloud deep learning.

More details see READMEs in each folder.

This project is still in progress, with ongoing work related to code organization and document writing.

# My Related Publications
Co-first Author, CIPR LCE 2024, under review.

	H.Fu, C.Wu, J.Kaiser, E.Barczakc, J.Pfrommerd, G.Lanzab, M.Heizmannc, J.Beyerera, "6D Pose Estimation on Point Cloud Data through Prior Knowledge Integration: A Case Study in Autonomous Disassembly"

Co-Author, CVPR 2024, submitted shortly (deadline: November 16th)

	C.Wu, K.Wang, Z.Zhong, H.Fu, J.Pfrommer, J.Beyerer, "Rethinking the Attention Module Design for Point Cloud Analysis"



# Structure of this Repository

```
.
├─ SFB_zivid/Haos_refactors		-- completed		-- controlling robot arm and Zivid camera with ROS, scanning single-view point cloud and generating full model point cloud 
├─ agi_para_extracter			-- completed		-- interface for other team members: extracting bolt position and further parameters of small motor
├─ alignment				-- completed		-- aligning 6D pose of full motor
├─ binary_segmentation			-- Document Sorting	-- pretraining and fine tuning for binary segmentation of single view model. DL model: PCT
├─ blender				-- Document Sorting	-- generating synthetic data set with blender 
├─ global_registration			-- Document Sorting	-- registering point cloud of full motor to the motor in single-view point cloud
├─ large_motor_segmentation		-- Document Sorting	-- pretraining and fine tuning for part segmentation of full model. DL model: PointNet, PointNet++, DGCNN, PCT
├─ pipeline				-- completed		-- full pipeline demo for the LCE paper
├─ Rethinking_Attention_Experiment	-- Project in Progress	-- Experiments for "Rethinking the Attention Module Design for Point Cloud Analysis"
├─ .gitattributes
├─ .gitignore
└─ README.md
```
