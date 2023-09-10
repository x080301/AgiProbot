def _pipline_merge_pdf(is_file_found=False):
    import PyPDF2
    import os

    dir = r'C:\Users\Lenovo\OneDrive\MV\print'
    merged_name = 'MV_merged.pdf'
    if is_file_found:

        merger = PyPDF2.PdfMerger()
        files = ['01-Introduction.pdf',
                 '02-Preprocessing.pdf',
                 '03-Edgedetection.pdf',
                 '04-Curvefitting.pdf',
                 '05-Color.pdf',
                 '06-Segmentation.pdf',
                 '07-Optics.pdf',
                 '10-Patternrecognition.pdf',
                 '12-DeepLearning.pdf']
        for filename in files:
            merger.append(PyPDF2.PdfReader(os.path.join(dir, filename)))
        merger.write(os.path.join(dir, merged_name))
    else:
        print(os.listdir(dir))


if __name__ == "__main__":
    _pipline_merge_pdf(True)
