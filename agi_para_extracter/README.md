# agi_para_extracter
Extract screw position and further parameters of small motor and its components from scaned pointcloud data(pcd) using deep learning

How to use:

    download folder "para_extracter" and use it as a python package

Interface summary:

        from para_extracter import ParaExtracter

        # necessary:
        extracter = ParaExtracter()                                             # Define the object
        extracter.load_model()                                                  # Load the trained model
        extracter.load_pcd_data(
            'D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')     # Load scaned pointcloud data
        extracter.run()                                                         # Run the model

        # optional: get what you need using following functions
        segementation_prediction = extracter.get_segmentation_prediction()
        classification_prediction = extracter.get_classification_prediction()

        if extracter.if_cover_existence(): 
            # Motor with a cover -> with cover screws, no visible gear. 
            # Motor without cover-> with visible gear, no cover screws.
            bolt_positions, cover_screw_normal, bolt_num, bolt_piont_clouds = extracter.find_screws()
        else:
            gear_piont_clouds, gearpositions = extracter.find_gears()

        # further data: just load further pcd file, run the model, and using funtions to get parameters.
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/B1_17_gear.pcd')
        extracter.run()
        segementation_prediction = extracter.get_segmentation_prediction()
        classification_prediction = extracter.get_classification_prediction()
