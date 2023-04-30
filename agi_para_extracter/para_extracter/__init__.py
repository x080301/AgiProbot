"""Generic (shallow and deep) copying operations.
Extract screw position and further parameters of small motor and its components from scaned pointcloud data(pcd) using deep learning

Interface summary:

        from para_extracter import ParaExtracter

        # necessary:
        extracter = ParaExtracter()                                             # Define the object
        extracter.load_model()                                                  # Dload the trained model
        extracter.load_pcd_data(
            'D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')     # load scaned pointcloud data
        extracter.run()                                                         # Run the model

        # optional: get what you need using following functions
        segementation_prediction = extracter.get_segmentation_prediction()
        classification_prediction = extracter.get_classification_prediction()

        if extracter.if_cover_existence(): # The motor can has a cover, which means no cover screw, or no cover, which means no visible gear.
            bolt_positions, cover_screw_normal, bolt_num, bolt_piont_clouds = extracter.find_screws()
        else:
            gear_piont_clouds, gearpositions = extracter.find_gears()

        # further data: just load the pcd file, run the model, and using funtions to get further parameters.
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/B1_17_gear.pcd')
        extracter.run()
        segementation_prediction = extracter.get_segmentation_prediction()
        classification_prediction = extracter.get_classification_prediction()

"""

from para_extracter import para_extracter

ParaExtracter = para_extracter.ParaExtracter
"""
define a class for parameter extracter, which is based on algorithm: Point Transformer

Args:
    None

Returns:
    None
"""
